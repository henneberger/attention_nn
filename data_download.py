"""Download GPT-2-tokenized FineWeb shards (modded-nanogpt format)."""

import os
import sys
import urllib.request

BASE = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main"
DEST = os.path.join(os.path.dirname(__file__), "data", "fineweb10B")


def get(fname):
    os.makedirs(DEST, exist_ok=True)
    path = os.path.join(DEST, fname)
    if os.path.exists(path):
        print(f"{fname}: already present")
        return
    print(f"downloading {fname} ...")
    urllib.request.urlretrieve(f"{BASE}/{fname}", path)


if __name__ == "__main__":
    num_train = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    get("fineweb_val_000000.bin")
    for i in range(1, num_train + 1):
        get("fineweb_train_%06d.bin" % i)
