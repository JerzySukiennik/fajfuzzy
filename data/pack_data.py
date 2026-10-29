"""Tokenise Fajfuzzy's corpora into flat uint16 binaries for random-window training.

Every document ends with <|sep|>. Every 500th document goes whole to validation, so no
text leaks across the split. With --delete each corpus file is removed once packed,
which keeps a Kaggle session inside its disk.

    python data/pack_data.py corpus_edu.txt ... --tokenizer tokenizer.json --out-prefix /kaggle/working/decide --delete
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tokenizer import iter_docs  # noqa: E402

BATCH_DOCS = 2000
VAL_EVERY = 500


def pack(paths, tok_path, prefix, delete):
    tok = Tokenizer.from_file(str(tok_path))
    assert tok.get_vocab_size() <= 65535, "vocab too large for uint16"
    sep = tok.token_to_id("<|sep|>")
    tf, vf = open(f"{prefix}_train.bin", "wb"), open(f"{prefix}_val.bin", "wb")
    counts = {"docs": 0, "train": 0, "val": 0, "per_source": {}}

    def flush(batch, name):
        for enc in tok.encode_batch(batch):
            ids = np.fromiter(enc.ids, dtype=np.uint16, count=len(enc.ids))
            ids = np.append(ids, np.uint16(sep))
            if counts["docs"] % VAL_EVERY == 0:
                ids.tofile(vf)
                counts["val"] += len(ids)
            else:
                ids.tofile(tf)
                counts["train"] += len(ids)
            counts["per_source"][name] = counts["per_source"].get(name, 0) + len(ids)
            counts["docs"] += 1
        batch.clear()

    for path in paths:
        name = path.stem.replace("corpus_", "")
        print(f"packing {name} ...", flush=True)
        batch = []
        for doc in iter_docs(path):
            batch.append(doc)
            if len(batch) >= BATCH_DOCS:
                flush(batch, name)
                if counts["docs"] % 200_000 < BATCH_DOCS:
                    print(f"  {counts['docs']:,} docs  {counts['train'] / 1e6:.0f}M train tokens", flush=True)
        flush(batch, name)
        if delete:
            path.unlink()
    tf.close()
    vf.close()
    print(json.dumps(counts, indent=1))
    Path(f"{prefix}_counts.json").write_text(json.dumps(counts))


def verify(prefix, tok_path):
    tok = Tokenizer.from_file(str(tok_path))
    arr = np.memmap(f"{prefix}_train.bin", dtype=np.uint16, mode="r")
    print(f"train tokens {len(arr):,}  max id {int(arr.max())}  vocab {tok.get_vocab_size()}")
    assert int(arr.max()) < tok.get_vocab_size()
    assert int((arr[:3_000_000] == tok.token_to_id("<|sep|>")).sum()) > 0, "no document separators"
    for frac in (0.1, 0.5, 0.9):
        s = int(len(arr) * frac)
        print(f"--- sample at {frac:.0%} ---")
        print(tok.decode([int(t) for t in arr[s:s + 100]], skip_special_tokens=False)[:400])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", type=Path)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    pack(a.corpus, a.tokenizer, a.out_prefix, a.delete)
    verify(a.out_prefix, a.tokenizer)
