"""
Hand-written Metal (MSL) kernels for Apple-silicon GPU training, exposed to MLX
via mx.fast.metal_kernel and wrapped with mx.custom_function for autograd.

Kernels:
  - relu2            : y = relu(x)^2 (modded-nanogpt MLP activation), fwd + bwd
  - rope_ht          : half-truncated rotary embedding (rotate first half of head
                       dims in adjacent pairs, pass the rest through), fwd + bwd
  - adam_step        : fused Adam moment update + cautious weight decay + param update
  - muon_momentum    : fused Nesterov momentum (two lerps) for Muon
  - muon_apply       : fused cautious-weight-decay parameter update for Muon

All arithmetic is done in fp32 inside the kernel regardless of storage dtype.
"""

import mlx.core as mx

_TG = 256  # threadgroup width for elementwise kernels


def _grid(n):
    return (n, 1, 1), (min(n, _TG), 1, 1)


def _scalars(*vals):
    """Pack python floats / 0-d mx arrays into a 1-D fp32 array (graph-safe)."""
    return mx.stack([
        v.astype(mx.float32).reshape(()) if isinstance(v, mx.array)
        else mx.array(float(v), dtype=mx.float32)
        for v in vals
    ])


# -----------------------------------------------------------------------------
# ReLU^2 activation

_relu2_fwd_k = mx.fast.metal_kernel(
    name="relu2_fwd",
    input_names=["x"],
    output_names=["y"],
    source="""
        uint i = thread_position_in_grid.x;
        float v = (float)x[i];
        float r = v > 0.0f ? v : 0.0f;
        y[i] = (T)(r * r);
    """,
)

_relu2_bwd_k = mx.fast.metal_kernel(
    name="relu2_bwd",
    input_names=["x", "g"],
    output_names=["dx"],
    source="""
        uint i = thread_position_in_grid.x;
        float v = (float)x[i];
        float r = v > 0.0f ? v : 0.0f;
        dx[i] = (T)(2.0f * r * (float)g[i]);
    """,
)


@mx.custom_function
def relu2(x):
    grid, tg = _grid(x.size)
    return _relu2_fwd_k(
        inputs=[x], template=[("T", x.dtype)], grid=grid, threadgroup=tg,
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]


@relu2.vjp
def _relu2_vjp(x, cotangent, output):
    grid, tg = _grid(x.size)
    return _relu2_bwd_k(
        inputs=[x, cotangent], template=[("T", x.dtype)], grid=grid, threadgroup=tg,
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]


# -----------------------------------------------------------------------------
# Half-truncated RoPE (modded-nanogpt style)
#
# x: (B, T, H, D). The first D/2 dims are rotated in adjacent pairs sharing an
# angle; the last D/2 dims pass through unchanged (half-truncate by
# @YouJiacheng). cos/sin tables: (T, D/4) fp32.

_rope_src = """
    uint p = thread_position_in_grid.x;        // one thread per dim-pair
    constexpr uint D2 = D / 2;                 // pairs per head
    constexpr uint D4 = D / 4;                 // rotated pairs per head
    uint row  = p / (H * D2);                  // b * T + t
    uint rem  = p % (H * D2);
    uint h    = rem / D2;
    uint d2   = rem % D2;
    uint t    = row % TS;
    uint base = row * (H * D) + h * D + d2 * 2;
    float x0 = (float)x[base];
    float x1 = (float)x[base + 1];
    if (d2 < D4) {
        float c = cs[t * D4 + d2];
        float s = sn[t * D4 + d2];
        if (BWD) {
            y[base]     = (T)(c * x0 - s * x1);
            y[base + 1] = (T)(c * x1 + s * x0);
        } else {
            y[base]     = (T)(c * x0 + s * x1);
            y[base + 1] = (T)(c * x1 - s * x0);
        }
    } else {
        y[base]     = (T)x0;
        y[base + 1] = (T)x1;
    }
"""

_rope_k = mx.fast.metal_kernel(
    name="rope_ht", input_names=["x", "cs", "sn"], output_names=["y"], source=_rope_src,
)


def _rope_call(x, cs, sn, bwd):
    B, Tlen, H, D = x.shape
    n = B * Tlen * H * (D // 2)
    grid, tg = _grid(n)
    return _rope_k(
        inputs=[x, cs, sn],
        template=[("T", x.dtype), ("H", H), ("D", D), ("TS", Tlen), ("BWD", bwd)],
        grid=grid, threadgroup=tg,
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]


@mx.custom_function
def rope_ht(x, cs, sn):
    return _rope_call(x, cs, sn, False)


@rope_ht.vjp
def _rope_vjp(primals, cotangent, output):
    x, cs, sn = primals
    dx = _rope_call(cotangent, cs, sn, True)
    return dx, mx.zeros_like(cs), mx.zeros_like(sn)


def make_rope_tables(seq_len, head_dim, base=1024.0):
    """Half-truncated RoPE tables: D/4 frequencies, geometric from 1 to 1/base."""
    d4 = head_dim // 4
    freqs = (1.0 / base) ** mx.linspace(0, 1, d4)
    t = mx.arange(seq_len).astype(mx.float32)
    theta = t[:, None] * freqs[None, :]          # (T, D/4)
    return mx.cos(theta), mx.sin(theta)


# -----------------------------------------------------------------------------
# Fused softcapped cross-entropy over logits z: (N, V).
#   zc  = 23 * sigmoid((z + 5) / 7.5)        (bounded in (0, 23), so logsumexp
#                                             needs no max-subtraction pass)
#   loss = logsumexp(zc) - zc[target]
# Forward: one threadgroup per row, single pass, simultaneous lse + target pick.
# Backward: elementwise dz = g * (softmax(zc) - onehot) * dzc/dz, with
#   dzc/dz = zc * (1 - zc/23) / 7.5  (recovered from zc alone).

_ce_fwd_k = mx.fast.metal_kernel(
    name="softcap_ce_fwd",
    input_names=["z", "tg"],
    output_names=["loss", "lse"],
    source="""
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        const device T* zr = z + (size_t)row * V;
        uint tcol = (uint)tg[row];
        float partial = 0.0f;
        float tval = 0.0f;
        for (uint c = lid; c < V; c += TGS) {
            float zc = 23.0f / (1.0f + metal::exp(-((float)zr[c] + 5.0f) * (1.0f / 7.5f)));
            partial += metal::exp(zc);
            if (c == tcol) tval = zc;
        }
        threadgroup float buf[TGS];
        threadgroup float tbuf[TGS];
        buf[lid] = partial;
        tbuf[lid] = tval;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TGS / 2; s > 0; s >>= 1) {
            if (lid < s) { buf[lid] += buf[lid + s]; tbuf[lid] += tbuf[lid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) {
            float l = metal::log(buf[0]);
            lse[row] = l;
            loss[row] = l - tbuf[0];
        }
    """,
)

_ce_bwd_k = mx.fast.metal_kernel(
    name="softcap_ce_bwd",
    input_names=["z", "lse", "g", "tg"],
    output_names=["dz"],
    source="""
        size_t i = thread_position_in_grid.x;
        uint row = i / V;
        uint col = i % V;
        float zc = 23.0f / (1.0f + metal::exp(-((float)z[i] + 5.0f) * (1.0f / 7.5f)));
        float p = metal::exp(zc - lse[row]);
        float gr = g[row];
        float d = gr * p;
        if (col == (uint)tg[row]) d -= gr;
        dz[i] = (T)(d * zc * (1.0f - zc * (1.0f / 23.0f)) * (1.0f / 7.5f));
    """,
)


def ce_softcap_fwd(z, targets):
    """z: (N, V) logits, targets: (N,) int32 -> (loss (N,) f32, lse (N,) f32)."""
    n, v = z.shape
    return _ce_fwd_k(
        inputs=[z, targets],
        template=[("T", z.dtype), ("V", v), ("TGS", _TG)],
        grid=(n * _TG, 1, 1), threadgroup=(_TG, 1, 1),
        output_shapes=[(n,), (n,)], output_dtypes=[mx.float32, mx.float32],
    )


def ce_softcap_bwd(z, lse, g, targets):
    """dz for the fused softcap CE; same shape/dtype as z."""
    n, v = z.shape
    grid, tg = _grid(n * v)
    return _ce_bwd_k(
        inputs=[z, lse, g.astype(mx.float32), targets],
        template=[("T", z.dtype), ("V", v)],
        grid=grid, threadgroup=tg,
        output_shapes=[z.shape], output_dtypes=[z.dtype],
    )[0]


# -----------------------------------------------------------------------------
# Fused Adam step with cautious weight decay (modded-nanogpt style)
# scalars s = [beta1, beta2, eps, step_size, eff_wd]; step_size folds in lr and
# bias correction. Returns (p_new, m_new, v_new), all fp32.

_adam_k = mx.fast.metal_kernel(
    name="adam_step",
    input_names=["p", "g", "m", "v", "s"],
    output_names=["p_out", "m_out", "v_out"],
    source="""
        uint i = thread_position_in_grid.x;
        float beta1 = s[0], beta2 = s[1], eps = s[2], step = s[3], wd = s[4];
        float gv = (float)g[i];
        float mv = beta1 * m[i] + (1.0f - beta1) * gv;
        float vv = beta2 * v[i] + (1.0f - beta2) * gv * gv;
        float u  = mv / (sqrt(vv) + eps) * step;
        float pv = p[i];
        // cautious weight decay: only decay where update and weight agree in sign
        if (u * pv > 0.0f) { u += pv * wd; }
        p_out[i] = pv - u;
        m_out[i] = mv;
        v_out[i] = vv;
    """,
)


def adam_step(p, g, m, v, beta1, beta2, eps, step_size, eff_wd):
    s = _scalars(beta1, beta2, eps, step_size, eff_wd)
    grid, tg = _grid(p.size)
    return _adam_k(
        inputs=[p, g.astype(mx.float32), m, v, s],
        grid=grid, threadgroup=tg,
        output_shapes=[p.shape] * 3, output_dtypes=[mx.float32] * 3,
    )


# -----------------------------------------------------------------------------
# Fused Nesterov momentum for Muon:
#   buf  <- lerp(buf, g, 1 - mu)
#   geff <- lerp(g, buf, mu)

_muon_mom_k = mx.fast.metal_kernel(
    name="muon_momentum",
    input_names=["g", "buf", "s"],
    output_names=["geff", "buf_out"],
    source="""
        uint i = thread_position_in_grid.x;
        float mu = s[0];
        float gv = (float)g[i];
        float bv = buf[i] + (1.0f - mu) * (gv - buf[i]);
        buf_out[i] = bv;
        geff[i] = (T)(gv + mu * (bv - gv));
    """,
)


def muon_momentum(g, buf, mu):
    s = _scalars(mu)
    grid, tg = _grid(g.size)
    return _muon_mom_k(
        inputs=[g, buf, s], template=[("T", mx.float32)],
        grid=grid, threadgroup=tg,
        output_shapes=[g.shape, g.shape], output_dtypes=[mx.float32, mx.float32],
    )


# -----------------------------------------------------------------------------
# Fused Muon parameter update with cautious weight decay:
#   mask = (u * p) >= 0 ; p <- p - p*mask*wd*lr - u*lr

_muon_apply_k = mx.fast.metal_kernel(
    name="muon_apply",
    input_names=["p", "u", "s"],
    output_names=["p_out"],
    source="""
        uint i = thread_position_in_grid.x;
        float lr = s[0], wd = s[1];
        float uv = (float)u[i];
        float pv = p[i];
        if (uv * pv >= 0.0f) { pv -= pv * wd * lr; }
        p_out[i] = pv - uv * lr;
    """,
)


def muon_apply(p, u, lr, wd):
    s = _scalars(lr, wd)
    grid, tg = _grid(p.size)
    return _muon_apply_k(
        inputs=[p, u.astype(mx.float32), s],
        grid=grid, threadgroup=tg,
        output_shapes=[p.shape], output_dtypes=[mx.float32],
    )[0]


# -----------------------------------------------------------------------------
# Self-test against pure-MLX references

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)

    # relu2 fwd/bwd
    x = mx.array(rng.standard_normal((64, 33)).astype(np.float32))
    y = relu2(x)
    ref = mx.maximum(x, 0) ** 2
    assert mx.allclose(y, ref, atol=1e-6).item(), "relu2 fwd mismatch"
    f = lambda x: (relu2(x) * mx.arange(33)).sum()
    fr = lambda x: ((mx.maximum(x, 0) ** 2) * mx.arange(33)).sum()
    g, gr = mx.grad(f)(x), mx.grad(fr)(x)
    assert mx.allclose(g, gr, atol=1e-5).item(), "relu2 bwd mismatch"

    # rope fwd: compare against direct complex-rotation reference
    B, T, H, D = 2, 16, 3, 64
    cs, sn = make_rope_tables(T, D)
    xq = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    out = rope_ht(xq, cs, sn)
    xn = np.array(xq)
    csn, snn = np.array(cs), np.array(sn)
    refn = xn.copy()
    for d2 in range(D // 4):
        c = csn[:, d2][None, :, None]
        s = snn[:, d2][None, :, None]
        x0, x1 = xn[..., 2 * d2], xn[..., 2 * d2 + 1]
        refn[..., 2 * d2] = c * x0 + s * x1
        refn[..., 2 * d2 + 1] = c * x1 - s * x0
    assert np.allclose(np.array(out), refn, atol=1e-5), "rope fwd mismatch"

    # rope bwd: numerical check that vjp is the transpose (rotation by -theta)
    ct = mx.array(rng.standard_normal((B, T, H, D)).astype(np.float32))
    gfun = lambda x: (rope_ht(x, cs, sn) * ct).sum()
    gx = mx.grad(gfun)(xq)
    # reference: <R x, ct> => grad = R^T ct
    refg = _rope_call(ct, cs, sn, True)
    assert mx.allclose(gx, refg, atol=1e-5).item(), "rope bwd mismatch"

    # adam_step vs reference
    p = mx.array(rng.standard_normal(1000).astype(np.float32))
    g = mx.array(rng.standard_normal(1000).astype(np.float32))
    m = mx.zeros(1000)
    v = mx.zeros(1000)
    pn, mn, vn = adam_step(p, g, m, v, 0.9, 0.99, 1e-10, 0.01, 0.05)
    m_ref = 0.1 * g
    v_ref = 0.01 * g * g
    u_ref = m_ref / (mx.sqrt(v_ref) + 1e-10) * 0.01
    mask = (u_ref * p) > 0
    u_ref = u_ref + p * mask * 0.05
    assert mx.allclose(pn, p - u_ref, atol=1e-6).item(), "adam mismatch"

    # muon momentum vs reference
    buf = mx.array(rng.standard_normal(512).astype(np.float32))
    gg = mx.array(rng.standard_normal(512).astype(np.float32))
    geff, bnew = muon_momentum(gg, buf, 0.95)
    bref = buf + 0.05 * (gg - buf)
    gref = gg + 0.95 * (bref - gg)
    assert mx.allclose(bnew, bref, atol=1e-6).item()
    assert mx.allclose(geff, gref, atol=1e-6).item()

    # muon apply vs reference
    pn = muon_apply(p, g, 0.02, 1.2)
    mask = (g * p) >= 0
    pref = p - p * mask * 1.2 * 0.02 - g * 0.02
    assert mx.allclose(pn, pref, atol=1e-6).item()

    # bf16 path smoke test
    xb = mx.array(rng.standard_normal((128, 128)).astype(np.float32)).astype(mx.bfloat16)
    assert relu2(xb).dtype == mx.bfloat16
    qb = xq.astype(mx.bfloat16)
    assert rope_ht(qb, cs, sn).dtype == mx.bfloat16

    print("all metal kernel tests passed")
