"""Run Fajfuzzy on the public JevBench items through JevBench's own runner and scorer
(github.com/fstandhartinger/jevbench), one request per decision, the way the board times
self-hosted models. Writes the runner's per-item records and the official summaries.

    python tools/jevbench_run.py --jevbench jevbench-repo --model decider.pt --tokenizer tokenizer.json --out run
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.decider import Decider, encode, option_probs  # noqa: E402
from train.finetune import jev_row  # noqa: E402

SPLITS = {"original": "standard", "easy": "easy", "hard": "hard"}


class FajfuzzyAdapter:
    name = "fajfuzzy"
    price_input_per_m = 0.0
    price_output_per_m = 0.0
    cost_basis = "self_hosted_compute_not_priced"

    def __init__(self, model_path, tokenizer_path, max_len, label):
        self.model = Decider.from_file(model_path).eval()
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.max_len = max_len
        self.label = label

    def reserve_estimate(self, task):
        return 0.0

    def decide(self, task):
        row = jev_row({"question": task.question, "labels": task.labels, "state": task.state,
                       "expected": task.expected}, "")
        ids = encode(self.tok, row["question"], row["state"], row["options"], None, self.max_len)
        x = torch.tensor([ids])
        with torch.inference_mode():
            p = option_probs(self.model(x, torch.tensor([len(ids)])), x)[0].tolist()
        labels = ["no", "yes"] if task.question["type"] == "noul" else task.labels
        return dict(zip(labels, p)), len(ids)

    def run(self, task):
        try:
            probs, n = self.decide(task)
        except ValueError as e:
            return self.result(False, error=f"input_rejected: {e}", status=422)
        return self.result(True, probs=probs, usage={"input_tokens": n, "output_tokens": 0}, status=200)

    def result(self, ok, **kw):
        from jevbench.adapters.base import DecisionResult
        status = kw.pop("status")
        return DecisionResult(self.name, ok, probs_source="native", model=self.label, status=status,
                              raw={"status": status}, request_body={"adapter": self.name}, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench", type=Path, required=True, help="clone of github.com/fstandhartinger/jevbench")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="first N items per split, for a smoke test")
    ap.add_argument("--name", default="fajfuzzy-v1")
    a = ap.parse_args()

    sys.path.insert(0, str(a.jevbench.resolve()))
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.summarize import summarize
    from jevbench.tasks import dataset_hash, load_jsonl

    torch.set_num_threads(a.threads or os.cpu_count() or 1)
    step = torch.load(str(a.model), map_location="cpu", weights_only=False).get("step")
    adapter = FajfuzzyAdapter(a.model, a.tokenizer, a.max_len, f"{a.name} ({a.model.name}, step {step})")
    files = {split: a.jevbench / "datasets" / "public" / f"{split}.jsonl" for split in SPLITS}
    tasks = {split: load_jsonl(str(path))[:a.limit or None] for split, path in files.items()}
    every = [t for split in SPLITS for t in tasks[split]]
    for t in every[:a.warmup]:
        adapter.decide(t)

    a.out.mkdir(parents=True, exist_ok=True)
    runner = Runner(adapter, Ledger(a.out / "ledger.jsonl", cap_usd=1), "/tmp/jevbench-raw")
    records = runner.run_all(every, progress_every=25, results_path=a.out / "records.jsonl")
    ids = {split: {t.id for t in tasks[split]} for split in SPLITS}
    summary = {
        "model": adapter.label,
        "max_len": a.max_len,
        "threads": torch.get_num_threads(),
        "host": {"cpu_count": os.cpu_count(), "machine": platform.machine(), "processor": platform.processor(),
                 "torch": torch.__version__},
        "dataset_hash": dataset_hash(every),
        "overall": summarize(every, records),
        "tiers": {SPLITS[s]: summarize(tasks[s], [r for r in records if r["task_id"] in ids[s]]) for s in SPLITS},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    o = summary["overall"]
    print(f"accuracy {o['accuracy']:.3f}  brier {o['brier_mean']:.3f}  ece {o['ece']['ece']:.3f}  "
          f"p50 {o['latency']['p50_s']:.3f} s  p95 {o['latency']['p95_s']:.3f} s", flush=True)
    for name, s in summary["tiers"].items():
        print(f"  {name:8s} {s['n_correct']}/{s['n_scorable']} = {s['accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
