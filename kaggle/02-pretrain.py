"""Kaggle kernel (T4 x2): one pretraining session of G-Decide, resumed from the previous one.

Pushed over and over by tools/chain.py. Kernel sources: gdec-prep (token binaries and
tokenizer) and, from session 2 on, the previous session (run1/ckpt.pt). The code is
pinned to the pretrain-v1 tag, so later work on main never touches a running schedule.

Kaggle can drop a kernel source and still start the run, which would then train from
random weights and quietly burn quota; EXPECT_RESUME turns that into a hard stop.
"""

import glob
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-decide.git"
TAG = "pretrain-v1"
SRC = "/tmp/g-decide"
OUT = "/kaggle/working/run1"

SESSION_HOURS = 10.5
EXPECT_RESUME = False

subprocess.run(["git", "clone", "--depth", "1", "--branch", TAG, REPO, SRC], check=True)
os.chdir(SRC)


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


data_train = find("decide_train.bin")
assert data_train, "decide_train.bin not found: attach gdec-prep as a kernel source"
prefix = data_train[: -len("_train.bin")]
print(f"data: {prefix}", flush=True)

os.makedirs(OUT, exist_ok=True)
tok = find("tokenizer.json")
if tok:
    shutil.copy(tok, f"{OUT}/tokenizer.json")

resume = find("ckpt.pt")
if EXPECT_RESUME and not resume:
    raise SystemExit("EXPECT_RESUME is set but no ckpt.pt was mounted: refusing to restart from step 0")

cmd = [sys.executable, "train/pretrain.py", "--data", prefix, "--out", OUT, "--max-hours", str(SESSION_HOURS)]
if resume:
    shutil.copy(resume, f"{OUT}/ckpt.pt")
    print(f"resuming from {resume}", flush=True)
    cmd.append("--resume")
else:
    print("session 1: training from scratch", flush=True)

subprocess.run(cmd, check=True)
print(f"checkpoint in {OUT}", flush=True)
