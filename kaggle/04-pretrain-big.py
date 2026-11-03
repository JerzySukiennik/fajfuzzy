"""Kaggle kernel (T4 x2): one pretraining session of the grown Fajfuzzy (24 layers), resumed from the previous one.

Pushed over and over by tools/chain_big.py, on whichever of the two accounts has quota. Inputs:
that account's token binaries (gdec-prep on the second account, fajfuzzy-big-prep on the
main one) and one starting point: the grown v1 (dataset fajfuzzy-big-init, session 1), the
previous session's output (same account) or its checkpoint carried over as the dataset
fajfuzzy-big-ckpt (other account). The code is pinned to the pretrain-v2 tag, and the
schedule below is frozen by session 1 (resumed sessions inherit it from the checkpoint).

EXPECT_STEP is the step the chain saw at the end of the previous session: a session that
finds no checkpoint, or an older one, stops instead of quietly retraining.
"""

import glob
import os
import shutil
import subprocess
import sys
import threading
import time

REPO = "https://github.com/JerzySukiennik/fajfuzzy.git"
TAG = "pretrain-v2"
SRC = "/tmp/fajfuzzy"
OUT = "/kaggle/working/run1"

SESSION_HOURS = 10.5
EXPECT_STEP = 0
SCHEDULE = ["--max-steps", "44000", "--warmup", "1000", "--lr", "3e-4", "--min-lr", "3e-5",
            "--batch-size", "32", "--grad-accum", "8", "--long-batch-size", "8", "--long-grad-accum", "8",
            "--block", "512", "--long-block", "2048", "--long-from", "0.92"]

subprocess.run(["git", "clone", "--depth", "1", "--branch", TAG, REPO, SRC], check=True)
os.chdir(SRC)


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


def warm(path):
    t = time.time()
    with open(path, "rb", buffering=0) as f:
        while f.read(64 << 20):
            pass
    print(f"data read once in {time.time() - t:.0f} s", flush=True)


data_train = find("decide_train.bin")
assert data_train, "decide_train.bin not found: attach the account's prep kernel as a source"
prefix = data_train[: -len("_train.bin")]
print(f"data: {prefix}", flush=True)

os.makedirs(OUT, exist_ok=True)
tok = find("tokenizer.json")
if tok:
    shutil.copy(tok, f"{OUT}/tokenizer.json")

cmd = [sys.executable, "train/pretrain.py", "--data", prefix, "--out", OUT, "--max-hours", str(SESSION_HOURS)]
if EXPECT_STEP:
    resume = find("ckpt.pt")
    if not resume:
        raise SystemExit(f"EXPECT_STEP is {EXPECT_STEP} but no ckpt.pt was mounted: refusing to restart")
    shutil.copy(resume, f"{OUT}/ckpt.pt")
    print(f"resuming from {resume}, expecting step {EXPECT_STEP} or later", flush=True)
    cmd += ["--resume", "--expect-step", str(EXPECT_STEP)]
else:
    init = find("grow_init.pt")
    if not init:
        raise SystemExit("session 1 needs grow_init.pt: attach the fajfuzzy-big-init dataset")
    print(f"session 1: starting from the grown v1 in {init}", flush=True)
    cmd += ["--init", init] + SCHEDULE

threading.Thread(target=warm, args=(data_train,), daemon=True).start()
subprocess.run(cmd, check=True, env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
print(f"checkpoint in {OUT}", flush=True)
