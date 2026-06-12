"""Sample from a trained checkpoint.

  python3 sample.py --prompt "The history of mathematics" --tokens 120
"""

import argparse

import mlx.core as mx

from model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoint.safetensors")
    ap.add_argument("--prompt", type=str, default="The")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        raise SystemExit("pip3 install --break-system-packages tiktoken")

    cfg = GPTConfig(num_layers=args.layers, num_heads=args.heads,
                    model_dim=args.dim,
                    long_layers=(args.layers // 2 - 1, args.layers - 1))
    model = GPT(cfg)
    path = args.ckpt
    if not path.endswith(".safetensors"):
        path = f"{path}.params.safetensors"
    params = mx.load(path)

    ids = enc.encode(args.prompt)
    print(args.prompt, end="", flush=True)
    for _ in range(args.tokens):
        ctx = ids[-cfg.seq_len:]
        z = model.logits(params, mx.array([ctx]))[0, -1]
        z = z[:50257] / args.temp
        if args.top_k:
            kth = mx.sort(z)[-args.top_k]
            z = mx.where(z < kth, mx.array(-1e9), z)
        nxt = int(mx.random.categorical(z).item())
        ids.append(nxt)
        print(enc.decode([nxt]), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
