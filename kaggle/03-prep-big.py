"""Kaggle kernel (CPU, Internet on, no GPU quota): token binaries for the grown Fajfuzzy on the main account.

Same sources as 01-prep.py, read from the other end (--reverse), so none of v1's shards come
back, and packed with v1's own tokenizer (from the fajfuzzy-decider-v1 dataset) so the
grown model keeps its vocabulary. Sessions on the second account train on v1's binaries,
sessions here on these: about 4.9B fresh tokens.
"""

import glob
import os
import shutil
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

tok =sorted(glob.glob("/kaggle/input/**/tokenizer.json", recursive=True))
assert tok, "tokenizer.json not found: attach the fajfuzzy-decider-v1 dataset"
shutil.copy(tok[0], f"{WORK}/tokenizer.json")

JOBS = [("edu", 13.0e9), ("web", 4.0e9), ("wiki", 3.0e9), ("html", 1.0e9)]

procs = {}
for name, chars in JOBS:
    out = f"{TMP}/corpus_{name}.txt"
    log = open(f"{TMP}/{name}.log", "w")
    procs[name] = (subprocess.Popen([sys.executable, "data/fetch_corpus.py", name, "--out", out, "--reverse",
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
subprocess.run([sys.executable, "data/pack_data.py", *corpora, "--tokenizer", f"{WORK}/tokenizer.json",
                "--out-prefix", f"{WORK}/decide", "--delete"], check=True)
print("done: decide_train.bin, decide_val.bin, tokenizer.json in", WORK)
