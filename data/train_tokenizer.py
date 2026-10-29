"""Train G-Decide's tokenizer: 32k byte-level BPE for English, every digit its own token.

The 32 special tokens come first (ids 0-31), so the model treats every id below 32 as
structure: padding, CLS and SEP, the ELECTRA mask and the decision markers used later
(state, question, criteria, options, one marker per option).

    python data/train_tokenizer.py corpus_edu.txt corpus_web.txt ... --out tokenizer.json
"""

import argparse
import random
import re
import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

DOC_SEP = "<|doc|>"
VOCAB = 32768
NAMED = ["<|pad|>", "<|cls|>", "<|sep|>", "<|mask|>", "<|state|>", "<|question|>",
         "<|criteria|>", "<|options|>", "<|opt|>"]
SPECIAL_TOKENS = NAMED + [f"<|reserved_{i}|>" for i in range(32 - len(NAMED))]

SPLIT = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
         r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+[ \t]*|\s+(?!\S)|\s+""")


def iter_docs(path: Path):
    buf = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.rstrip("\n") == DOC_SEP:
                doc = "".join(buf)
                buf.clear()
                if doc.strip():
                    yield doc
            else:
                buf.append(line)


def sample(paths, budget_per_file, seed=0):
    rng = random.Random(seed)
    for path in paths:
        used = 0
        for doc in iter_docs(path):
            if rng.random() > 0.35:
                continue
            yield doc[:20000]
            used += min(len(doc), 20000)
            if used >= budget_per_file:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("tokenizer.json"))
    ap.add_argument("--chars-per-file", type=float, default=150e6)
    a = ap.parse_args()

    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  min_frequency=2, show_progress=False)
    tok.train_from_iterator(sample(a.corpus, a.chars_per_file), trainer)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(a.out))

    multi = [t for t in tok.get_vocab() if re.fullmatch(r"\s?\d{2,}", t)]
    print(f"vocab {tok.get_vocab_size()}, multi-digit tokens {len(multi)} (must be 0)")
    assert not multi, multi[:10]
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"{t} has id {tok.token_to_id(t)}, expected {i}"
    tests = ["Is a hotdog a sub?",
             "<|cls|><|question|>Which intent does the message express?<|state|>When will my order "
             "arrive?<|options|><|opt|>track_order<|opt|>cancel_order<|sep|>",
             "[button] Sign in   [link] Forgot password?   [textbox] Email",
             "Loss $22,400 less the $1,000 deductible, capped at $15,000 (HE-2291)."]
    for s in tests:
        ids = tok.encode(s).ids
        assert tok.decode(ids, skip_special_tokens=False) == s, f"roundtrip failed: {s!r}"
        print(f"{len(ids):3d} tok / {len(s):3d} ch  {s[:60]!r}")


if __name__ == "__main__":
    sys.exit(main())
