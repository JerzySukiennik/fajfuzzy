"""Turn a JevBench run (tools/jevbench_run.py) into the board's own axes, as close as the public
material allows, and place it among the published systems.

Intelligence and Calibration follow the board's v1.6 definitions, checked digit for digit against
its published aggregate: per type (choice, noul, score) a tier-weighted chance-corrected accuracy
(easy 0.1, standard 0.2, judge 0.3, hard 0.4) and Intelligence their mean; Calibration the mean of
choice 100(1 - ECE/0.5) on hard items, noul 100(1 - ECE/0.5) and score the mean of 100(1 - ECE/0.5)
and 100(1 - NRPS). Capability is mean(Intelligence, Calibration).

The board's own 300 public and 1,200 sealed items are not published, so this runs on the 231
public items of the scorer repo, which has no judge tier. The repo also keeps every listed system's
right/wrong answer on these same 231 items, which gives an exact like-for-like Intelligence and,
through the systems that are on both, a fitted estimate of the board Intelligence.

    python tools/jevbench_score.py --run run --jevbench jevbench-repo --board board-live.json --out bench.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TIER_WEIGHTS = {"easy": 0.1, "standard": 0.2, "judge": 0.3, "hard": 0.4}
SPLITS = {"original": "standard", "easy": "easy", "hard": "hard"}
TYPES = ("choice", "noul", "score")
GPU_USD_PER_HOUR, LOAD, MAX_TOKENS_PER_S = 1.5235, 0.70, 100_000
COST_CAP_USD, LATENCY_CAP_S = 0.06459465517241379, 1.2329566404223442


def load_tasks(repo):
    tasks = {}
    for split, tier in SPLITS.items():
        for line in open(Path(repo) / "datasets" / "public" / f"{split}.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                tasks[r["id"]] = {"tier": tier, "type": r["question"]["type"], "labels": [str(x) for x in r["labels"]],
                                  "expected": str(r["expected"])}
    return tasks


def intelligence(tasks, correct):
    cells = defaultdict(lambda: [0, 0, 0.0])
    for tid, t in tasks.items():
        c = cells[(t["type"], t["tier"])]
        c[0] += bool(correct.get(tid))
        c[1] += 1
        c[2] += 1 / len(t["labels"])
    per_type, tiers = {}, {}
    for typ in TYPES:
        s = w = 0.0
        for tier, tw in TIER_WEIGHTS.items():
            if (typ, tier) in cells:
                hits, n, chance = cells[(typ, tier)]
                cc = 100 * (hits / n - chance / n) / (1 - chance / n)
                tiers[f"{typ}|{tier}"] = {"n": n, "correct": hits, "accuracy": hits / n, "cc": cc}
                s, w = s + tw * cc, w + tw
        per_type[typ] = s / w
    return max(0.0, sum(per_type.values()) / len(per_type)), per_type, tiers


def ece(pairs, bins=10):
    total, err = len(pairs), 0.0
    for b in range(bins):
        cell = [(c, ok) for c, ok in pairs if min(int(c * bins), bins - 1) == b]
        if cell:
            err += len(cell) / total * abs(sum(ok for _, ok in cell) / len(cell) - sum(c for c, _ in cell) / len(cell))
    return err


def nrps(probs, labels, expected):
    k = len(labels)
    cum = np.cumsum([probs[label] for label in labels])[:-1]
    truth = np.array([float(i >= labels.index(expected)) for i in range(k - 1)])
    return float(((cum - truth) ** 2).sum() / (k - 1))


def calibration(tasks, records):
    pick = lambda typ, tier=None: [(tasks[r["task_id"]], r) for r in records
                                   if tasks[r["task_id"]]["type"] == typ and (tier is None or tasks[r["task_id"]]["tier"] == tier)]
    pairs = lambda rows: [(max(r["probs"].values()), r["correct"]) for _, r in rows if r.get("probs")]
    score_of = lambda e: max(0.0, 100 * (1 - e / 0.5))
    choice = pick("choice", "hard")
    noul = pick("noul")
    score = pick("score")
    parts = {
        "choice": {"ece_hard": ece(pairs(choice)), "n": len(choice)},
        "noul": {"ece": ece(pairs(noul)), "n": len(noul)},
        "score": {"top_ece": ece(pairs(score)), "n": len(score),
                  "nrps": float(np.mean([nrps(r["probs"], t["labels"], t["expected"]) for t, r in score]))},
    }
    parts["choice"]["score"] = score_of(parts["choice"]["ece_hard"])
    parts["noul"]["score"] = score_of(parts["noul"]["ece"])
    parts["score"]["score"] = (score_of(parts["score"]["top_ece"]) + 100 * (1 - parts["score"]["nrps"])) / 2
    return sum(p["score"] for p in parts.values()) / len(parts), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="folder with records.jsonl and summary.json")
    ap.add_argument("--jevbench", type=Path, required=True, help="clone of github.com/fstandhartinger/jevbench")
    ap.add_argument("--board", type=Path, required=True, help="board rows scraped from benchmarkheaven.com/jev-models")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    tasks = load_tasks(a.jevbench)
    records = [json.loads(line) for line in open(a.run / "records.jsonl") if line.strip()]
    summary = json.loads((a.run / "summary.json").read_text())
    i_pub, per_type, tiers = intelligence(tasks, {r["task_id"]: r["correct"] for r in records})
    c_pub, parts = calibration(tasks, records)

    per_task = json.loads((a.jevbench / "results" / "v1.2" / "jevbench-v1.2-per-task.json").read_text())
    board = json.loads(a.board.read_text())
    by_key = {r["key"]: r for r in board}
    same = []
    for key, s in per_task["systems"].items():
        if s["partial"]:
            continue
        i231, pt, _ = intelligence(tasks, {tid: v[0] == "c" for tid, v in s["public_tasks"].items()})
        row = by_key.get(key)
        same.append({"key": key, "display": s["display"], "i_same_items": i231, "per_type": pt,
                     "board_intelligence": row["intelligence"] if row else None,
                     "board_capability": row["capability"] if row else None, "cat": row["cat"] if row else None})

    fit = [(x["i_same_items"], x["board_intelligence"]) for x in same if x["board_intelligence"] is not None]
    x, y = np.array(fit).T
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sd = float(math.sqrt((resid ** 2).sum() / (len(x) - 2)))
    se = sd * math.sqrt(1 + 1 / len(x) + (i_pub - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    i_est = min(100.0, max(0.0, slope * i_pub + intercept))
    cap_est = (i_est + c_pub) / 2

    o = summary["overall"]
    p50 = o["latency"]["p50_s"]
    tokens = float(np.mean([r["usage"]["input_tokens"] for r in records if r.get("usage")]))
    usd = GPU_USD_PER_HOUR * tokens * 1000 / (3600 * MAX_TOKENS_PER_S * LOAD)
    ranked = [r for r in board if r["rank"] != "–"]
    out = {
        "model": summary["model"],
        "items": len(records),
        "official_summary": {"accuracy": o["accuracy"], "brier_mean": o["brier_mean"], "ece": o["ece"]["ece"],
                             "ordinal_mae": o["ordinal_mae"], "schema_validity": o["schema_validity"],
                             "tiers": {k: {"accuracy": v["accuracy"], "n": v["n_scorable"]} for k, v in summary["tiers"].items()}},
        "public_items": {"intelligence": i_pub, "per_type": per_type, "tiers": tiers,
                         "calibration": c_pub, "calibration_parts": parts, "capability": (i_pub + c_pub) / 2},
        "board_estimate": {"intelligence": i_est, "intelligence_band": [max(0.0, i_est - se), min(100.0, i_est + se)],
                           "calibration": c_pub, "capability": cap_est,
                           "capability_band": [(max(0.0, i_est - se) + c_pub) / 2, (min(100.0, i_est + se) + c_pub) / 2],
                           "fit": {"slope": slope, "intercept": intercept, "r": float(np.corrcoef(x, y)[0, 1]),
                                   "residual_sd": sd, "n": len(x)},
                           "rank_all": 1 + sum(r["capability"] > cap_est for r in board),
                           "rank_jev_class": 1 + sum(r["capability"] > cap_est for r in ranked),
                           "n_all": len(board), "n_jev_class": len(ranked)},
        "latency": {"p50_measured_s": p50, "p95_measured_s": o["latency"]["p95_s"], "p50_adjusted_s": 2 * p50 + 0.15,
                    "within_latency_cap": 2 * p50 + 0.15 <= LATENCY_CAP_S, "host": summary["host"],
                    "threads": summary["threads"]},
        "cost": {"usd_per_1000_estimate": usd, "tokens_per_decision": tokens,
                 "cost_score": max(0.0, min(100.0, 100 - 30 * math.log10(usd / 0.001))),
                 "within_cost_cap": usd <= COST_CAP_USD,
                 "basis": f"GPU-time rule, USD {GPU_USD_PER_HOUR}/h at {LOAD:.0%} load, throughput at the {MAX_TOKENS_PER_S:,} tokens/s cap"},
        "same_items": sorted(same + [{"key": "fajfuzzy-v1", "display": summary["model"], "i_same_items": i_pub,
                                      "per_type": per_type, "board_intelligence": None, "board_capability": None,
                                      "cat": "fajfuzzy"}], key=lambda r: -r["i_same_items"]),
    }
    a.out.write_text(json.dumps(out, indent=1))
    b = out["board_estimate"]
    print(f"public items: I {i_pub:.1f} ({', '.join(f'{k} {v:.1f}' for k, v in per_type.items())}), C {c_pub:.1f} "
          f"({', '.join(f'{k} {v['score']:.1f}' for k, v in parts.items())})")
    print(f"board estimate: I {i_est:.1f} [{b['intelligence_band'][0]:.1f}, {b['intelligence_band'][1]:.1f}], "
          f"Cap {cap_est:.1f}, rank {b['rank_all']}/{b['n_all']} (Jev-class {b['rank_jev_class']}/{b['n_jev_class']}); "
          f"fit r {b['fit']['r']:.2f} on {b['fit']['n']} systems")
    print(f"latency p50 {p50:.3f} s -> adjusted {2 * p50 + 0.15:.2f} s, cost ~${usd:.4f}/1k")
    jev = next(r for r in out["same_items"] if r["key"] == "jev-1.13.0")
    print(f"same 231 items: Fajfuzzy {i_pub:.1f} vs Jev 1.13.0 {jev['i_same_items']:.1f}; place "
          f"{[r['key'] for r in out['same_items']].index('fajfuzzy-v1') + 1}/{len(out['same_items'])}")


if __name__ == "__main__":
    main()
