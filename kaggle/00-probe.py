"""Kaggle kernel (T4 x2, ~15 min): measure Fajfuzzy's real pretraining throughput.

Runs the actual training step (DataParallel, fp16, generator plus discriminator) on random
tokens at 512 and 2048 tokens of context for every micro-batch that fits, at the run's
131,072 tokens per optimiser step. The numbers set the step budget of the real run.
"""

import json
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/fajfuzzy.git"
SRC = "/tmp/fajfuzzy"
TOKENS_PER_STEP = 131072

subprocess.run(["git", "clone", "--depth", "1", REPO, SRC], check=True)
os.chdir(SRC)

results = []
for block, sizes in ((512, (128, 96, 64, 48, 32)), (2048, (32, 24, 16, 8))):
    for size in sizes:
        accum = max(1, TOKENS_PER_STEP // (size * block))
        r = subprocess.run([sys.executable, "train/pretrain.py", "--random-data", "--probe", "--max-steps", "8",
                            "--block", str(block), "--long-from", "1.0", "--batch-size", str(size),
                            "--grad-accum", str(accum), "--out", "/tmp/probe"],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("PROBE ")]
        if r.returncode == 0 and line:
            res = json.loads(line[-1][6:])
            results.append(res)
            print(f"context {block:>4}  batch {size:>3} x {accum}  {res['tok_per_s']:>7,} tok/s  "
                  f"peak {res['peak_gb']} GB", flush=True)
        else:
            oom = "out of memory" in (r.stderr + r.stdout).lower()
            print(f"context {block:>4}  batch {size:>3}: {'OOM' if oom else 'failed'}", flush=True)
            if not oom:
                print(r.stdout[-1500:], r.stderr[-2500:], flush=True)

with open("/kaggle/working/probe.json", "w") as f:
    json.dump(results, f, indent=1)
for block in (512, 2048):
    best = max((r for r in results if r["block"] == block), key=lambda r: r["tok_per_s"], default=None)
    print(f"best at {block}: {best}")
