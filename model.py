"""
Attention-based LM for Apple-silicon GPU (MLX), architecture following
modded-nanogpt:

  - untied embedding / lm_head, zero-init output projections, normal(0.005) head
  - RMS norms (no learnable weights), QK-norm
  - half-truncated RoPE (custom Metal kernel), attn scale 0.1
  - ReLU^2 MLP (custom Metal kernel)
  - learnable per-layer residual / post / x0-injection lambdas, and per-layer
    QKV/O weight scale lambdas (sa_lambdas)
  - sliding-window attention on most layers, full causal on a few
  - softcapped logits 23*sigmoid((z+5)/7.5) with a memory-efficient chunked
    cross-entropy (recompute-in-backward custom_function)

Parameters live in a flat dict of fp32 master arrays; compute is bf16.
Muon-bank layout: attn_bank (L,4,d,d) [q,k,v,o], fc_bank (L,d,4d),
proj_bank (L,4d,d) -- all stored (in, out) so forward is x @ W.
"""

import math
from dataclasses import dataclass

import mlx.core as mx

from metal_kernels import make_rope_tables, relu2, rope_ht

BF16 = mx.bfloat16


@dataclass
class GPTConfig:
    vocab_size: int = 50304        # GPT-2 50257 padded to multiple of 128
    num_layers: int = 8
    num_heads: int = 8
    model_dim: int = 512
    seq_len: int = 1024
    window: int = 384              # sliding window size for short layers
    long_layers: tuple = (3, 7)    # layers with full causal attention
    attn_scale: float = 0.1
    rope_base: float = 1024.0
    ce_chunk: int = 2048           # rows per lm_head/CE chunk

    @property
    def head_dim(self):
        return self.model_dim // self.num_heads


def rms(x):
    return mx.fast.rms_norm(x, None, 1e-6)


def init_params(cfg: GPTConfig, seed=0):
    mx.random.seed(seed)
    d, L = cfg.model_dim, cfg.num_layers
    std = 0.5 * d ** -0.5
    bound = (3 ** 0.5) * std

    def u(*shape):
        return mx.random.uniform(-bound, bound, shape, dtype=mx.float32)

    head = mx.random.normal((d, cfg.vocab_size), dtype=mx.float32) * 0.005
    attn_bank = mx.stack([
        mx.stack([u(d, d), u(d, d), u(d, d), mx.zeros((d, d))])  # q, k, v, o(zero)
        for _ in range(L)
    ])
    params = {
        "embed": head.T,                            # tied init, untied training
        "lm_head": head,
        "attn_bank": attn_bank,                     # (L, 4, d, d)
        "fc_bank": mx.stack([u(d, 4 * d) for _ in range(L)]),
        "proj_bank": mx.zeros((L, 4 * d, d)),       # zero-init c_proj
        "resid_lambdas": mx.full((L, 2), 1.1 ** 0.5),
        "post_lambdas": mx.ones((L, 2)),
        "x0_lambdas": mx.zeros((L,)),
        "sa_lambdas": mx.tile(mx.array([0.5, 1.0]), (L, 1)),
    }
    return params


def optimizer_table():
    return {
        "embed":         {"optim": "adam", "adam_betas": (0.5, 0.95), "wd_mul": 150.0},
        "lm_head":       {"optim": "adam", "adam_betas": (0.5, 0.95), "wd_mul": 150.0},
        "attn_bank":     {"optim": "muon"},
        "fc_bank":       {"optim": "muon"},
        "proj_bank":     {"optim": "muon"},
        "resid_lambdas": {"optim": "adam", "adam_betas": (0.9, 0.95), "lr_mul": 5.0, "wd_mul": 0.0},
        "post_lambdas":  {"optim": "adam", "adam_betas": (0.9, 0.95), "lr_mul": 5.0, "wd_mul": 0.0},
        "x0_lambdas":    {"optim": "adam", "adam_betas": (0.9, 0.95), "lr_mul": 5.0, "wd_mul": 0.0},
        "sa_lambdas":    {"optim": "adam", "adam_betas": (0.9, 0.95), "lr_mul": 5.0, "wd_mul": 0.0},
    }


# -----------------------------------------------------------------------------
# Softcapped cross-entropy, fused via custom Metal kernels (metal_kernels.py).
# Forward computes softcap + logsumexp + target-pick in one pass over bf16
# logits; the logits are carried to the backward (as an extra output) so the
# vjp is one elementwise kernel + two matmuls -- no recompute, no fp32
# (N, V) intermediates.

from metal_kernels import ce_softcap_bwd, ce_softcap_fwd


def _softcap_chunk_loss(xc, w, tc):
    """Pure-MLX reference (used in tests)."""
    z = mx.matmul(xc, w).astype(mx.float32)
    zc = 23.0 * mx.sigmoid((z + 5.0) / 7.5)
    lse = mx.logsumexp(zc, axis=-1)
    tl = mx.take_along_axis(zc, tc[:, None], axis=-1).squeeze(-1)
    return lse - tl


@mx.custom_function
def _softcap_ce_raw(xf, w, targets):
    z = mx.matmul(xf, w)
    loss, lse = ce_softcap_fwd(z, targets)
    return loss, z, lse


@_softcap_ce_raw.vjp
def _softcap_ce_vjp(primals, cotangents, outputs):
    xf, w, targets = primals
    _, z, lse = outputs
    g = cotangents[0]
    dz = ce_softcap_bwd(z, lse, g, targets)
    dx = mx.matmul(dz, mx.swapaxes(w, 0, 1))
    dw = mx.matmul(mx.swapaxes(xf, 0, 1), dz)
    return dx, dw


def softcap_ce(xf, w, targets, chunk=None):
    return _softcap_ce_raw(xf, w, targets)[0]


# -----------------------------------------------------------------------------
# Forward

class GPT:
    def __init__(self, cfg: GPTConfig):
        self.cfg = cfg
        cs, sn = make_rope_tables(cfg.seq_len, cfg.head_dim, cfg.rope_base)
        self.rope_cs, self.rope_sn = cs, sn
        # Block-banded attention: split queries into blocks; each block only
        # multiplies against key blocks inside its (causal, windowed) band.
        # ~30% less score area than masked full T^2, and a much cheaper
        # backward than the decomposed sdpa vjp.
        self.qblock = 512
        self.blocks_window = self._make_blocks(cfg.seq_len, cfg.window)
        self.blocks_causal = self._make_blocks(cfg.seq_len, None)
        self.layer_blocks = [
            self.blocks_causal if l in cfg.long_layers else self.blocks_window
            for l in range(cfg.num_layers)
        ]

    def _make_blocks(self, T, window):
        idx = mx.arange(T)
        blocks = []
        for qs in range(0, T, self.qblock):
            qe = min(qs + self.qblock, T)
            ks = 0 if window is None else max(0, qs - window)
            qi = idx[qs:qe][:, None]
            ki = idx[ks:qe][None, :]
            keep = qi >= ki
            if window is not None:
                keep = keep & (qi - ki < window)
            am = mx.where(keep, mx.array(0.0, dtype=BF16),
                          mx.array(-30000.0, dtype=BF16))
            blocks.append((qs, qe, ks, am))
        return blocks

    def _attention(self, q, k, v, blocks, scale):
        """q,k,v: (B, H, T, Dh) bf16."""
        outs = []
        for qs, qe, ks, am in blocks:
            s = mx.matmul(q[:, :, qs:qe], mx.swapaxes(k[:, :, ks:qe], -2, -1))
            p = mx.softmax(s * scale + am, axis=-1, precise=True)
            outs.append(mx.matmul(p, v[:, :, ks:qe]))
        return mx.concatenate(outs, axis=2)

    def __call__(self, params, tokens, targets):
        """tokens, targets: (B, T) int32. Returns mean loss."""
        cfg = self.cfg
        B, T = tokens.shape
        H, Dh, d = cfg.num_heads, cfg.head_dim, cfg.model_dim

        resid = params["resid_lambdas"].astype(BF16)
        post = params["post_lambdas"].astype(BF16)
        x0l = params["x0_lambdas"].astype(BF16)
        sal = params["sa_lambdas"].astype(BF16)

        x = rms(params["embed"][tokens].astype(BF16))
        x0 = x

        for l in range(cfg.num_layers):
            wl = params["attn_bank"][l].astype(BF16)        # (4, d, d)
            qkv_w = mx.concatenate([wl[0], wl[1], wl[2]], axis=-1) * sal[l, 0]
            wo = wl[3] * sal[l, 1]

            h = rms(x)
            qkv = mx.matmul(h, qkv_w).reshape(B, T, 3, H, Dh)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = rms(q), rms(k)                            # QK norm
            q = rope_ht(q, self.rope_cs, self.rope_sn)
            k = rope_ht(k, self.rope_cs, self.rope_sn)
            y = self._attention(
                q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
                self.layer_blocks[l], cfg.attn_scale,
            )
            y = y.transpose(0, 2, 1, 3).reshape(B, T, d)
            y = mx.matmul(y, wo)
            x = resid[l, 0] * x + post[l, 0] * y + x0l[l] * x0

            h = rms(x)
            y = mx.matmul(relu2(mx.matmul(h, params["fc_bank"][l].astype(BF16))),
                          params["proj_bank"][l].astype(BF16))
            x = resid[l, 1] * x + post[l, 1] * y

        x = rms(x).reshape(B * T, d)
        losses = softcap_ce(x, params["lm_head"].astype(BF16),
                            targets.reshape(-1), cfg.ce_chunk)
        return losses.mean()

    def logits(self, params, tokens):
        """Inference path: (B, T) -> (B, T, V) softcapped logits, T <= seq_len."""
        cfg = self.cfg
        B, T = tokens.shape
        H, Dh, d = cfg.num_heads, cfg.head_dim, cfg.model_dim

        resid = params["resid_lambdas"].astype(BF16)
        post = params["post_lambdas"].astype(BF16)
        x0l = params["x0_lambdas"].astype(BF16)
        sal = params["sa_lambdas"].astype(BF16)
        cs, sn = self.rope_cs[:T], self.rope_sn[:T]
        idx = mx.arange(T)

        x = rms(params["embed"][tokens].astype(BF16))
        x0 = x
        for l in range(cfg.num_layers):
            wl = params["attn_bank"][l].astype(BF16)
            qkv_w = mx.concatenate([wl[0], wl[1], wl[2]], axis=-1) * sal[l, 0]
            wo = wl[3] * sal[l, 1]
            h = rms(x)
            qkv = mx.matmul(h, qkv_w).reshape(B, T, 3, H, Dh)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k = rms(q), rms(k)
            q = rope_ht(q, cs, sn)
            k = rope_ht(k, cs, sn)
            keep = idx[:, None] >= idx[None, :]
            if l not in cfg.long_layers:
                keep = keep & (idx[:, None] - idx[None, :] < cfg.window)
            y = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
                scale=cfg.attn_scale, mask=keep,
            )
            y = mx.matmul(y.transpose(0, 2, 1, 3).reshape(B, T, d), wo)
            x = resid[l, 0] * x + post[l, 0] * y + x0l[l] * x0
            h = rms(x)
            y = mx.matmul(relu2(mx.matmul(h, params["fc_bank"][l].astype(BF16))),
                          params["proj_bank"][l].astype(BF16))
            x = resid[l, 1] * x + post[l, 1] * y

        z = mx.matmul(rms(x), params["lm_head"].astype(BF16)).astype(mx.float32)
        return 23.0 * mx.sigmoid((z + 5.0) / 7.5)


if __name__ == "__main__":
    import numpy as np

    # softcap_ce gradient check vs unchunked reference
    rng = np.random.default_rng(0)
    N, d, V = 96, 32, 257
    xf = mx.array(rng.standard_normal((N, d)).astype(np.float32))
    w = mx.array(rng.standard_normal((d, V)).astype(np.float32) * 0.05)
    tg = mx.array(rng.integers(0, V, N).astype(np.int32))

    def fused(xf, w):
        return softcap_ce(xf, w, tg, 40).mean()

    def ref(xf, w):
        return _softcap_chunk_loss(xf, w, tg).mean()

    lf, gf = mx.value_and_grad(fused, argnums=(0, 1))(xf, w)
    lr_, gr = mx.value_and_grad(ref, argnums=(0, 1))(xf, w)
    assert abs(lf.item() - lr_.item()) < 1e-4, "ce loss mismatch"
    assert mx.allclose(gf[0], gr[0], atol=1e-5).item(), "ce dx mismatch"
    assert mx.allclose(gf[1], gr[1], atol=1e-5).item(), "ce dw mismatch"
    print("softcap_ce gradcheck passed")

    # tiny model forward/backward smoke test
    cfg = GPTConfig(vocab_size=512, num_layers=2, num_heads=4, model_dim=64,
                    seq_len=32, window=16, long_layers=(1,), ce_chunk=32)
    params = init_params(cfg)
    model = GPT(cfg)
    toks = mx.array(rng.integers(0, 512, (2, 32)).astype(np.int32))
    tgts = mx.array(rng.integers(0, 512, (2, 32)).astype(np.int32))
    loss, grads = mx.value_and_grad(lambda p: model(p, toks, tgts))(params)
    mx.eval(loss, grads)
    print(f"tiny model loss: {loss.item():.4f} (expect ~ln(512)={math.log(512):.4f})")
    for k, v in grads.items():
        gn = mx.sqrt(mx.sum(v.astype(mx.float32) ** 2)).item()
        assert math.isfinite(gn), f"non-finite grad for {k}"
        print(f"  grad {k:14s} norm {gn:.4f}")
    print("model smoke test passed")
