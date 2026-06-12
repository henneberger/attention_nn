"""
Muon (NorMuon variant) + Adam optimizer for MLX, following modded-nanogpt:

  - Nesterov momentum (fused Metal kernel) -> Polar Express orthogonalization
    (batched bf16 matmuls) -> NorMuon low-rank variance reduction -> cautious
    weight decay update (fused Metal kernel) for stacked 2D parameter banks.
  - Adam with bias correction + cautious weight decay (single fused Metal
    kernel) for embeddings and scalars.

Polar Express: https://arxiv.org/pdf/2505.16932 (coefficients for num_iters=5,
safety_factor=2e-2, cushion=2, from modded-nanogpt).
NorMuon: https://arxiv.org/pdf/2510.05491
"""

import mlx.core as mx

from metal_kernels import adam_step, muon_apply, muon_momentum

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(g):
    """Orthogonalize a bank of matrices g: (..., M, N), bf16 matmuls."""
    x = g.astype(mx.float32)
    # spectral norm upper bound: frobenius norm (with safety cushion)
    n = mx.sqrt(mx.sum(x * x, axis=(-2, -1), keepdims=True))
    x = (x / (n * 1.02 + 1e-6)).astype(mx.bfloat16)
    tall = x.shape[-2] > x.shape[-1]
    for a, b, c in POLAR_EXPRESS_COEFFS:
        if tall:
            A = mx.matmul(mx.swapaxes(x, -2, -1), x)   # (.., N, N)
            B = b * A + c * mx.matmul(A, A)
            x = a * x + mx.matmul(x, B)
        else:
            A = mx.matmul(x, mx.swapaxes(x, -2, -1))   # (.., M, M)
            B = b * A + c * mx.matmul(A, A)
            x = a * x + mx.matmul(B, x)
    return x


def normuon_rescale(v, second_mom, beta2):
    """NorMuon variance reduction; v: (..., M, N) bf16, second_mom fp32.

    Returns (rescaled v fp32, new second_mom). Algebra mirrors
    modded-nanogpt's _apply_normuon_variance_reduction.
    """
    M, N = v.shape[-2], v.shape[-1]
    red_dim = -1 if M >= N else -2
    vf = v.astype(mx.float32)
    v_mean = mx.mean(vf * vf, axis=red_dim, keepdims=True)
    red_size = v.shape[red_dim]
    v_norm = mx.sqrt(mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_size)
    second_mom = second_mom + (1 - beta2) * (v_mean - second_mom)
    step_size = mx.rsqrt(mx.maximum(second_mom, 1e-10))
    scaled_sq = (v_mean * red_size) * mx.square(step_size)
    v_norm_new = mx.sqrt(mx.sum(scaled_sq, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
    return vf * final_scale, second_mom


class MuonAdam:
    """Combined NorMuon + Adam optimizer over a flat {name: param} dict.

    table[name] = {"optim": "muon"|"adam", and per-param overrides:
                   "lr_mul", "wd_mul", "adam_betas"}
    Muon params must be stacked banks of 2D matrices (shape (..., M, N)).
    """

    def __init__(self, params, table, muon_lr=0.023, muon_wd=1.2, muon_beta2=0.9,
                 adam_lr=0.008, adam_wd=0.005, adam_eps=1e-10):
        self.table = table
        self.muon_lr = muon_lr
        self.muon_wd = muon_wd
        self.muon_beta2 = muon_beta2
        self.adam_lr = adam_lr
        self.adam_wd = adam_wd
        self.adam_eps = adam_eps
        # step counter lives in state so a compiled step can trace its increment
        self.state = {"_step": {"t": mx.zeros(())}}
        for name, p in params.items():
            cfg = table[name]
            if cfg["optim"] == "muon":
                M, N = p.shape[-2], p.shape[-1]
                red_shape = list(p.shape)
                red_shape[-1 if M >= N else -2] = 1
                self.state[name] = {
                    "momentum": mx.zeros(p.shape, dtype=mx.float32),
                    "second_mom": mx.zeros(red_shape, dtype=mx.float32),
                }
                # shape-based lr multiplier (tall matrices get sqrt(M/N))
                cfg["shape_mul"] = max(1.0, M / N) ** 0.5
            else:
                self.state[name] = {
                    "m": mx.zeros(p.shape, dtype=mx.float32),
                    "v": mx.zeros(p.shape, dtype=mx.float32),
                }

    def __call__(self, params, grads, lr_scale=1.0, momentum=0.95):
        """Functional step: returns dict of updated params (fp32).

        lr_scale and momentum may be python floats or 0-d mx arrays (the
        latter keeps a compiled step from retracing as schedules change).
        """
        if not isinstance(lr_scale, mx.array):
            lr_scale = mx.array(float(lr_scale))
        t = self.state["_step"]["t"] + 1.0
        self.state["_step"]["t"] = t
        new_params = {}
        for name, p in params.items():
            cfg = self.table[name]
            g = grads[name]
            st = self.state[name]
            if cfg["optim"] == "muon":
                lr = self.muon_lr * cfg.get("lr_mul", 1.0) * cfg["shape_mul"] * lr_scale
                eff_wd = cfg.get("wd_mul", 1.0) * self.muon_wd * self.muon_lr * lr_scale
                g_eff, st["momentum"] = muon_momentum(g.astype(mx.float32), st["momentum"], momentum)
                u = polar_express(g_eff)
                u, st["second_mom"] = normuon_rescale(u, st["second_mom"], self.muon_beta2)
                new_params[name] = muon_apply(p, u, lr, eff_wd)
            else:
                beta1, beta2 = cfg.get("adam_betas", (0.9, 0.95))
                lr = self.adam_lr * cfg.get("lr_mul", 1.0) * lr_scale
                bias1 = 1 - mx.power(mx.array(beta1), t)
                bias2 = 1 - mx.power(mx.array(beta2), t)
                step_size = lr * mx.sqrt(bias2) / bias1
                eff_wd = lr * lr * self.adam_wd * cfg.get("wd_mul", 1.0)
                pn, st["m"], st["v"] = adam_step(
                    p, g, st["m"], st["v"], beta1, beta2, self.adam_eps, step_size, eff_wd
                )
                new_params[name] = pn
        return new_params

    def state_arrays(self):
        """All optimizer state arrays (for mx.eval)."""
        out = []
        for st in self.state.values():
            out.extend(st.values())
        return out

    def save_state(self, path):
        flat = {f"{name}|{k}": v for name, st in self.state.items()
                for k, v in st.items()}
        mx.save_safetensors(path, flat)

    def load_state(self, path):
        flat = mx.load(path)
        for key, v in flat.items():
            name, k = key.rsplit("|", 1)
            assert name in self.state and k in self.state[name], f"unknown state {key}"
            assert self.state[name][k].shape == v.shape, f"shape mismatch for {key}"
            self.state[name][k] = v


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)

    # polar express should produce (approximately) a semi-orthogonal matrix
    for shape in [(4, 256, 64), (4, 64, 256), (2, 512, 512)]:
        g = mx.array(rng.standard_normal(shape).astype(np.float32))
        u = polar_express(g)
        M, N = shape[-2], shape[-1]
        if M >= N:
            gram = mx.matmul(mx.swapaxes(u, -2, -1), u).astype(mx.float32)
            eye = mx.eye(N)
        else:
            gram = mx.matmul(u, mx.swapaxes(u, -2, -1)).astype(mx.float32)
            eye = mx.eye(M)
        err = mx.max(mx.abs(gram - eye)).item()
        print(f"polar_express {shape}: max |U^T U - I| = {err:.3f}")
        assert err < 0.35, "orthogonalization too loose"

    # end-to-end optimizer smoke test: minimize ||W x - y||^2 on a fixed batch
    d_in, d_out, nb = 64, 32, 256
    X = mx.array(rng.standard_normal((nb, d_in)).astype(np.float32))
    Wtrue = mx.array(rng.standard_normal((d_in, d_out)).astype(np.float32))
    Y = X @ Wtrue

    params = {
        "w": mx.zeros((1, d_in, d_out)),
        "b": mx.zeros((d_out,)),
    }
    table = {
        "w": {"optim": "muon"},
        "b": {"optim": "adam", "adam_betas": (0.9, 0.95)},
    }
    opt = MuonAdam(params, table, muon_lr=0.05, adam_lr=0.01)

    def loss_fn(ps):
        pred = X @ ps["w"][0] + ps["b"]
        return mx.mean((pred - Y) ** 2)

    lg = mx.value_and_grad(loss_fn)
    l0 = loss_fn(params).item()
    for i in range(200):
        loss, grads = lg(params)
        params = opt(params, grads, lr_scale=1.0, momentum=0.9)
        mx.eval(params)
    l1 = loss_fn(params).item()
    print(f"muon+adam regression: loss {l0:.3f} -> {l1:.4f}")
    assert l1 < l0 * 0.01, "optimizer failed to reduce loss"
    print("muon optimizer tests passed")
