"""Kaggle kernel (CPU, Internet on, no GPU quota): build Fajfuzzy's tokenizer and token binaries.

Four fetchers run in parallel because the network is the bottleneck. Text lands in /tmp,
so the saved output holds only decide_train.bin, decide_val.bin, decide_counts.json and
tokenizer.json. A failed source is skipped rather than fatal. Character budgets aim at
~5.3B tokens, a little more than two weeks of T4 x2 can consume, so no text repeats.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/fajfuzzy.git"
WORK = "/kaggle/working"
TMP = "/tmp/fajfuzzy"
SRC = f"{TMP}/fajfuzzy"
os.makedirs(TMP, exist_ok=True)

subprocess.run(["git", "clone", "--depth", "1", REPO, SRC], check=True)
os.chdir(SRC)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "pyarrow"], check=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

JOBS = [("edu", 12.0e9), ("web", 5.0e9), ("wiki", 4.5e9), ("html", 1.3e9)]

procs = {}
for name, chars in JOBS:
    out = f"{TMP}/corpus_{name}.txt"
    log = open(f"{TMP}/{name}.log", "w")
    procs[name] = (subprocess.Popen([sys.executable, "data/fetch_corpus.py", name, "--out", out,
                                     "--max-chars", str(chars)], stdout=log, stderr=subprocess.STDOUT), out)

corpora = []
for name, (p, out) in procs.items():
    rc = p.wait()
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"{name}: exit {rc}, {size / 1e9:.2f} GB", flush=True)
    print(open(f"{TMP}/{name}.log").read()[-800:], flush=True)
    if size > 50e6:
        corpora.append(out)

assert corpora, "no source produced data"
subprocess.run([sys.executable, "data/train_tokenizer.py", *corpora, "--out", f"{WORK}/tokenizer.json",
                "--chars-per-file", "150e6"], check=True)
subprocess.run([sys.executable, "data/pack_data.py", *corpora, "--tokenizer", f"{WORK}/tokenizer.json",
                "--out-prefix", f"{WORK}/decide", "--delete"], check=True)
print("done: decide_train.bin, decide_val.bin, tokenizer.json in", WORK)
