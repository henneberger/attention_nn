# attention_nn — Muon-trained attention LM for Apple-silicon GPU

A modded-nanogpt-style GPT trained with the **Muon optimizer**, implemented in
[MLX](https://github.com/ml-explore/mlx) with **hand-written Metal (MSL)
kernels** (no Objective-C — kernels are MSL source compiled at runtime via
`mx.fast.metal_kernel`). Tuned for an M3 Pro (18-core GPU, 18 GB unified
memory).

## Files

| file | contents |
|---|---|
| `metal_kernels.py` | Custom MSL kernels + autograd wrappers: ReLU², half-truncated RoPE (fwd/bwd), fused softcapped cross-entropy (fwd reduction + bwd), fused Adam step, fused Muon Nesterov momentum, fused cautious-weight-decay update. Self-tests vs MLX references. |
| `muon.py` | NorMuon + Adam: Polar Express orthogonalization (batched bf16 matmuls), NorMuon variance reduction, shape-scaled LR, cautious weight decay. Self-tests. |
| `model.py` | The transformer (flat param dict, fp32 masters / bf16 compute) + fused CE wiring. Self-tests incl. CE gradcheck. |
| `train.py` | Data loader (modded-nanogpt `.bin` shards), LR cooldown + Muon momentum schedules, fully `mx.compile`d train step. |

## Architecture (from modded-nanogpt)

- untied embedding / lm_head, zero-init output projections (o_proj, c_proj),
  normal(0.005) head, uniform `±√3·0.5/√d` inputs
- RMS norms without learnable weights; **QK-norm**
- **half-truncated RoPE** (first half of head dims rotated, rest stationary),
  attention scale 0.1
- **ReLU² MLP** (4×)
- learnable per-layer residual/post/x0-injection lambdas (`resid` init √1.1)
  and QKV/O weight-scale lambdas (init 0.5 / 1.0)
- **sliding-window attention** (window 384) on most layers, full causal on
  layers {L/2−1, L−1}, implemented **block-banded** (queries in blocks of 512,
  out-of-band key blocks skipped — ~30% cheaper than masked full T²)
- logits softcap `23·σ((z+5)/7.5)`, cross-entropy fused into 2 Metal kernels
- Muon (Polar Express, 5 iters, bf16) on attention/MLP banks; Adam on
  embed/lm_head/lambdas; momentum warmup 0.85→0.95, linear LR cooldown to 0.15

Omitted from modded-nanogpt (multi-GPU/speedrun-specific or heavy):
FP8 matmuls, value embeddings, MTP, MUDD, bigram hash embeddings, YaRN window
extension schedule, paired heads, attention gates.

## Performance notes (M3 Pro, 18 GB)

End-to-end throughput went **800 → ~9,000 tok/s** (76.7M params, T=1024,
batch 12):

1. **`mx.set_cache_limit(2 GB)`** — the single biggest fix. MLX's buffer cache
   grew past 9 GB and pushed macOS into swap; steps went 1.9s → 40s. Cap it.
2. **Fused softcap CE** — peak memory 10 GB → 5.2 GB (no fp32 (N,V)
   intermediates; bf16 logits saved for backward, no recompute).
3. **`mx.compile` of the whole step** (fwd+bwd+optimizer). Schedule scalars
   enter as 0-d `mx.array`s so changing lr/momentum doesn't retrace.
4. **Block-banded attention** — MLX's sdpa backward is a decomposed full-T²
   fallback; banding cuts each windowed layer's fwd+bwd 23ms → 17ms.
5. Batch 12 is the sweet spot: 16 re-enters memory pressure on 18 GB.

## Run

```bash
python3 data_download.py          # or: curl the two shards (see below)
python3 metal_kernels.py && python3 muon.py && python3 model.py   # self-tests
python3 train.py --steps 2000     # small config (76.7M), ~45 min on M3 Pro
```

### Full GPT-2-scale run (modded-nanogpt dimensions)

155M params (d=768, 11 layers, 6 heads, head_dim 128), ~0.75B FineWeb tokens.
~4,300 tok/s on M3 Pro → roughly 2 days. Checkpoints (params + optimizer
state) are written every `--val-every` steps; `--resume` continues exactly.

```bash
python3 data_download.py 8        # 800M train tokens
nohup caffeinate -i python3 train.py \
  --steps 91500 --batch 8 --dim 768 --layers 11 --heads 6 \
  --val-every 500 --log-every 50 --muon-lr 0.015 --adam-lr 0.005 \
  --save gpt2_ckpt > gpt2_run.log 2>&1 &

tail -f gpt2_run.log              # watch progress
# after interruption: add --resume to the same command
python3 sample.py --ckpt gpt2_ckpt.params.safetensors --dim 768 --layers 11 --heads 6 \
  --prompt "The history of"      # sample from latest checkpoint
```

LRs are scaled down from modded-nanogpt's (0.023/0.008) because the per-step
batch here is 8K tokens vs their 131K.

Data: GPT-2-tokenized FineWeb shards (modded-nanogpt format) in
`data/fineweb10B/`:

```
https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin
https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_train_000001.bin
```
