"""
Train the modded-nanogpt-style attention LM with Muon on Apple-silicon GPU.

Usage:
  python3 train.py                      # default config
  python3 train.py --steps 2000 --batch 8 --seq 1024

Data: GPT-2 token shards in data/fineweb10B/*.bin (modded-nanogpt format:
256 int32 header [magic, version, num_tokens], then uint16 tokens).
"""

import argparse
import glob
import math
import time

import mlx.core as mx
import numpy as np

from model import GPT, GPTConfig, init_params, optimizer_table
from muon import MuonAdam


# -----------------------------------------------------------------------------
# Data

def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, "magic number mismatch"
    num_tokens = int(header[2])
    tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4,
                       shape=(num_tokens,))
    return tokens


def batch_iterator(shards, batch_size, seq_len, start_tokens=0):
    """Sequential batches of (inputs, targets), each (B, T) int32.

    shards: list of token memmaps, cycled in order. start_tokens fast-forwards
    (for resume) without loading data.
    """
    if not isinstance(shards, (list, tuple)):
        shards = [shards]
    span = batch_size * seq_len
    sizes = [len(s) - span - 1 for s in shards]
    total = sum(sizes)
    offset = start_tokens % total
    si = 0
    while offset >= sizes[si]:
        offset -= sizes[si]
        si += 1
    pos = offset
    while True:
        tokens = shards[si]
        if pos + span + 1 >= len(tokens):
            si = (si + 1) % len(shards)
            pos = 0
            tokens = shards[si]
        buf = np.asarray(tokens[pos:pos + span + 1]).astype(np.int32)
        yield (mx.array(buf[:-1].reshape(batch_size, seq_len)),
               mx.array(buf[1:].reshape(batch_size, seq_len)))
        pos += span


# -----------------------------------------------------------------------------
# Schedules (modded-nanogpt style)

def lr_scale(step, total_steps, cooldown_frac=0.6, final=0.15):
    cd_start = int(total_steps * (1 - cooldown_frac))
    if step < cd_start:
        return 1.0
    t = min(1.0, (step - cd_start) / max(1, total_steps - cd_start))
    return 1.0 * (1 - t) + final * t


def muon_momentum_sched(step, total_steps, warmup=300, cooldown=50,
                        lo=0.85, hi=0.95):
    if step < warmup:
        return lo + (hi - lo) * step / warmup
    cd_start = total_steps - cooldown
    if step > cd_start:
        return hi - (hi - lo) * (step - cd_start) / cooldown
    return hi


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--window", type=int, default=384)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--muon-lr", type=float, default=0.023)
    ap.add_argument("--adam-lr", type=float, default=0.008)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--save", type=str, default="checkpoint",
                    help="checkpoint path prefix")
    ap.add_argument("--resume", action="store_true",
                    help="resume from --save checkpoint")
    ap.add_argument("--max-shards", type=int, default=99)
    ap.add_argument("--cache-gb", type=float, default=2.0,
                    help="MLX buffer cache cap; too high causes swap on 18GB machines")
    args = ap.parse_args()

    mx.set_cache_limit(int(args.cache_gb * 2**30))

    cfg = GPTConfig(num_layers=args.layers, num_heads=args.heads,
                    model_dim=args.dim, seq_len=args.seq, window=args.window,
                    long_layers=(args.layers // 2 - 1, args.layers - 1))
    model = GPT(cfg)
    params = init_params(cfg, seed=args.seed)
    opt = MuonAdam(params, optimizer_table(),
                   muon_lr=args.muon_lr, adam_lr=args.adam_lr)

    n_params = sum(p.size for p in params.values())
    print(f"model: {cfg.num_layers}L d={cfg.model_dim} h={cfg.num_heads} "
          f"T={cfg.seq_len} window={cfg.window} | {n_params/1e6:.1f}M params")

    train_files = sorted(glob.glob("data/fineweb10B/fineweb_train_*.bin"))[:args.max_shards]
    val_files = sorted(glob.glob("data/fineweb10B/fineweb_val_*.bin"))
    assert train_files and val_files, "run data download first"
    train_shards = [load_shard(f) for f in train_files]
    val_tokens = load_shard(val_files[0])
    print(f"train tokens: {sum(len(s) for s in train_shards)/1e6:.0f}M "
          f"({len(train_shards)} shards)  val tokens: {len(val_tokens)/1e6:.0f}M")

    # ---- resume ----
    import json
    import os
    start_step = 0
    if args.resume and os.path.exists(f"{args.save}.meta.json"):
        with open(f"{args.save}.meta.json") as f:
            meta = json.load(f)
        start_step = meta["step"]
        loaded = mx.load(f"{args.save}.params.safetensors")
        for k in params:
            assert params[k].shape == loaded[k].shape
        params = dict(loaded)
        opt.load_state(f"{args.save}.opt.safetensors")
        print(f"resumed from step {start_step}")

    train_it = batch_iterator(train_shards, args.batch, cfg.seq_len,
                              start_tokens=start_step * args.batch * cfg.seq_len)
    val_batches = []
    vit = batch_iterator(val_tokens, args.batch, cfg.seq_len)
    for _ in range(args.val_batches):
        val_batches.append(next(vit))

    loss_and_grad = mx.value_and_grad(lambda p, x, y: model(p, x, y))

    # Train step compiled with params + optimizer state as captured state.
    train_state = {"params": params, "opt": opt.state}

    def step_fn(x, y, lr_s, mom):
        loss, grads = loss_and_grad(train_state["params"], x, y)
        train_state["params"] = opt(train_state["params"], grads, lr_s, mom)
        return loss

    if args.no_compile:
        train_step = step_fn
    else:
        train_step = mx.compile(step_fn, inputs=[train_state], outputs=[train_state])

    def val_loss():
        total = 0.0
        for x, y in val_batches:
            total += model(train_state["params"], x, y).item()
        return total / len(val_batches)

    def save_checkpoint(step):
        if not args.save:
            return
        mx.eval(train_state)
        mx.save_safetensors(f"{args.save}.params.safetensors", train_state["params"])
        opt.save_state(f"{args.save}.opt.safetensors")
        with open(f"{args.save}.meta.json", "w") as f:
            json.dump({"step": step, "config": vars(args)}, f, default=str)

    tokens_per_step = args.batch * cfg.seq_len
    t_start = time.perf_counter()
    trained_tokens = start_step * tokens_per_step
    window_t0, window_tokens = t_start, 0

    for step in range(start_step, args.steps + 1):
        if step % args.val_every == 0 or step == args.steps:
            mx.eval(train_state)
            vl = val_loss()
            elapsed = time.perf_counter() - t_start
            print(f"step {step:5d} | val_loss {vl:.4f} | "
                  f"{trained_tokens/1e6:.2f}M tokens | {elapsed:.0f}s | "
                  f"peak_mem {mx.get_peak_memory()/2**30:.2f}GB", flush=True)
            if step > start_step:
                save_checkpoint(step)
            if step == args.steps:
                break
            window_t0, window_tokens = time.perf_counter(), 0

        x, y = next(train_it)
        loss = train_step(x, y, mx.array(lr_scale(step, args.steps)),
                          mx.array(muon_momentum_sched(step, args.steps)))
        mx.async_eval(loss, train_state)
        trained_tokens += tokens_per_step
        window_tokens += tokens_per_step

        if (step + 1) % args.log_every == 0:
            lv = loss.item()  # sync point
            dt = time.perf_counter() - window_t0
            tps = window_tokens / dt
            print(f"step {step+1:5d} | loss {lv:.4f} | {tps:,.0f} tok/s | "
                  f"lr {lr_scale(step, args.steps):.2f}", flush=True)
            if not math.isfinite(lv):
                raise SystemExit("loss diverged (non-finite)")

    print(f"done. total time {time.perf_counter()-t_start:.0f}s, "
          f"{trained_tokens/1e6:.1f}M tokens")
    if args.save:
        save_checkpoint(args.steps)
        print(f"saved checkpoint to {args.save}.*")


if __name__ == "__main__":
    main()
