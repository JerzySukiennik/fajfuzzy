"""Fetch Fajfuzzy's English pretraining corpus into text files of documents.

  edu   HuggingFaceFW/fineweb-edu sample-10BT   educational prose, the core (ODC-By)
  web   HuggingFaceFW/fineweb sample-10BT       everyday web text: forums, shops, reviews (ODC-By)
  wiki  wikimedia/wikipedia 20231101.en         encyclopedic facts (CC BY-SA 4.0)
  html  common-pile/stackv2_html_filtered       page markup, for reading UI state (Blue Oak licenses)

Whole parquet shards download in the background while the previous one is filtered,
which is far faster on Kaggle than streaming. Documents are separated by a line holding
only DOC_SEP, exact duplicates (ignoring whitespace) are dropped, and finished shards are
recorded so a rerun resumes.

    python data/fetch_corpus.py edu --out /tmp/fajfuzzy/corpus_edu.txt --max-chars 12e9
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DOC_SEP = "<|doc|>"
HF = "https://huggingface.co"
PARQUET = {
    "edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT", 300),
    "web": ("HuggingFaceFW/fineweb", "sample/10BT", 300),
    "wiki": ("wikimedia/wikipedia", "20231101.en", 500),
}
HTML_SHARDS = [f"{HF}/datasets/common-pile/stackv2_html_filtered/resolve/main/stackv2_html-{i:04d}.json.gz"
               for i in range(5)]
WIKI_TAIL = re.compile(r"\n(References|See also|External links|Further reading|Notes|Bibliography)\n.*\Z", re.S)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def headers():
    h = {"User-Agent": "fajfuzzy-prep"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


class Sink:
    def __init__(self, out: Path, max_chars: float):
        self.out, self.max_chars = out, max_chars
        self.done_file = Path(f"{out}.done")
        self.done = set(self.done_file.read_text().split()) if self.done_file.exists() else set()
        self.seen = set()
        self.chars = out.stat().st_size if out.exists() else 0
        self.f = open(out, "a", encoding="utf-8")
        self.kept = self.dupes = 0

    def full(self):
        return self.chars >= self.max_chars

    def add(self, text):
        text = text.strip()
        if not text:
            return
        h = hashlib.md5(re.sub(r"\s+", "", text).encode("utf-8", "ignore")).digest()
        if h in self.seen:
            self.dupes += 1
            return
        self.seen.add(h)
        text = text.replace("<|", "< |")
        self.f.write(text + f"\n{DOC_SEP}\n")
        self.chars += len(text)
        self.kept += 1

    def mark_done(self, key):
        self.f.flush()
        self.done.add(key)
        self.done_file.write_text("\n".join(sorted(self.done)))

    def close(self):
        self.f.close()


def download(url, path):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers()), timeout=120) as r, \
                    open(f"{path}.part", "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            os.replace(f"{path}.part", path)
            return True
        except Exception as e:
            log(f"retry {attempt + 1} {path.name}: {e}")
            time.sleep(10 * (attempt + 1))
    return False


def list_parquet(repo, folder):
    url = f"{HF}/api/datasets/{repo}/tree/main/{folder}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers()), timeout=60) as r:
        return sorted(x["path"] for x in json.load(r) if x["path"].endswith(".parquet"))


def fetch_parquet(source, sink):
    import pyarrow.parquet as pq
    repo, folder, min_chars = PARQUET[source]
    files = list_parquet(repo, folder)
    random.Random(7).shuffle(files)
    files = [f for f in files if f not in sink.done]
    raw = Path(f"{sink.out}.raw")
    raw.mkdir(exist_ok=True)

    def local(f):
        return raw / f.replace("/", "_")

    def fetch(f):
        return download(f"{HF}/datasets/{repo}/resolve/main/{f}", local(f))

    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(fetch, files[0]) if files else None
        for i, f in enumerate(files):
            ok = pending.result()
            pending = pool.submit(fetch, files[i + 1]) if i + 1 < len(files) and not sink.full() else None
            if ok:
                for batch in pq.ParquetFile(local(f)).iter_batches(columns=["text"], batch_size=4096):
                    for text in batch.column(0).to_pylist():
                        if text and len(text) >= min_chars:
                            sink.add(WIKI_TAIL.sub("", text) if source == "wiki" else text)
                    if sink.full():
                        break
                local(f).unlink()
            sink.mark_done(f)
            log(f"  {f}: kept {sink.kept:,} dupes {sink.dupes:,} total {sink.chars / 1e9:.2f}G chars")
            if sink.full():
                break
        if pending is not None:
            pending.result()
    shutil.rmtree(raw, ignore_errors=True)


def keep_html(text, path):
    n = len(text)
    if n < 500 or n > 60_000 or re.search(r"(node_modules|/vendor/|/dist/|\.min\.|bundle)", path or "", re.I):
        return False
    lines = text.split("\n")
    if max(len(l) for l in lines) > 1000:
        return False
    return text.count("base64,") <= 3


def fetch_html(sink):
    for url in HTML_SHARDS:
        key = url.rsplit("/", 1)[-1]
        if key in sink.done or sink.full():
            continue
        for _ in range(4):
            try:
                r = urllib.request.urlopen(urllib.request.Request(url, headers=headers()), timeout=120)
                with io.TextIOWrapper(gzip.GzipFile(fileobj=r), encoding="utf-8", errors="ignore") as g:
                    for line in g:
                        if sink.full():
                            break
                        try:
                            d = json.loads(line)
                        except ValueError:
                            continue
                        m = d.get("metadata", {})
                        if keep_html(d.get("text", ""), m.get("path") or m.get("filename") or ""):
                            sink.add(d["text"])
                break
            except Exception as e:
                log(f"stream broke: {e}")
                time.sleep(15)
        sink.mark_done(key)
        log(f"  {key}: kept {sink.kept:,} dupes {sink.dupes:,} total {sink.chars / 1e9:.2f}G chars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=list(PARQUET) + ["html"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-chars", type=float, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sink = Sink(a.out, a.max_chars)
    log(f"{a.source}: target {a.max_chars / 1e9:.2f}G chars, already {sink.chars / 1e9:.2f}G")
    if a.source == "html":
        fetch_html(sink)
    else:
        fetch_parquet(a.source, sink)
    sink.close()
    log(f"{a.source}: finished at {sink.chars / 1e9:.2f}G chars")
