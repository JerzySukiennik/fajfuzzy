"""Build Fajfuzzy's fine-tuning data (runs on a Kaggle CPU kernel): train.jsonl.gz and val.jsonl.gz
in the layout train/finetune.py reads.

Two parts:
- tasksource-jev-typed-decisions (default mix, ~1M rows): typed decisions from hundreds of
  human-annotated tasks, soft targets where annotators disagreed. Dropped: non-English rows,
  tasks built from Claude, GPT-4 or Gemini outputs (their terms forbid training competing
  models on them), and any row sharing a 13-word run with a JevBench state.
- GUIrilla-Task (macOS): real accessibility trees crawled from 1,100 apps. Only the trees and
  the clicked element are used; the GPT-4-written instructions are not. Each click becomes a
  step instruction built from the element itself ("Click the OK button"), the way Gemini will
  hand single steps to Fajfuzzy in Concierge, with up to 12 on-screen elements as options. Two
  more labelled controls from the same window become extra steps (source ...-extra).

    python data/prep_finetune.py --jev jevbench-public --out prep
"""

import argparse
import ast
import gzip
import hashlib
import json
import random
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download, list_repo_files

TS = "tasksource/tasksource-jev-typed-decisions"
GUI = "macpaw-research/GUIrilla-Task"
TS_COLS = ["state", "kind", "options", "target", "question", "source", "variant", "license_use"]
GUI_COLS = ["app_name", "action", "accessibility", "element_data"]
EXCLUDE_DATASETS = {
    "Anthropic/persuasion", "tasksource/hh-rlhf", "lmsys/chatbot_arena_conversations",
    "pushpdeep/UltraFeedback-paired", "TIGER-Lab/WebInstruct-verified", "tasksource/webinstruct",
    "tasksource/synthetic-retrieval-NLI", "openai/summarize_from_feedback",
    "vwxyzjn/summarize_from_feedback_oai_preprocessing", "openai/webgpt_comparisons",
}
EXCLUDE_PREFIXES = ("multilingual/", "webgpt_comparisons", "summarize_from_feedback")
NGRAM = 13
ACTIONABLE = {"AXButton", "AXRadioButton", "AXCheckBox", "AXPopUpButton", "AXMenuButton", "AXMenuItem",
              "AXMenuBarItem", "AXTextField", "AXTextArea", "AXComboBox", "AXLink", "AXSlider",
              "AXDisclosureTriangle", "AXIncrementor", "AXSearchField", "AXSegmentedControl", "AXCell", "AXRow"}
VALUE_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField", "AXStaticText"}
TEMPLATES = {
    "button": ["Click the {L} button", "Press {L}", "Click {L}", "Hit the {L} button"],
    "tab": ["Open the {L} tab", "Switch to the {L} tab", "Click the {L} tab"],
    "radio button": ["Select {L}", "Choose the {L} option", "Click the {L} radio button"],
    "checkbox": ["Toggle {L}", "Click the {L} checkbox", "Tick {L}"],
    "text field": ["Click the {L} field", "Focus the {L} text field", "Click into {L}"],
    "link": ["Open the {L} link", "Click the {L} link"],
    "menu item": ["Choose {L} from the menu", "Click the {L} menu item"],
    "menu bar item": ["Open the {L} menu", "Click {L} in the menu bar"],
    "pop up button": ["Open the {L} pop-up menu", "Click the {L} dropdown"],
}
FALLBACK = ["Click {L}", "Click the {L} {R}"]


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text):
    w = words(text)
    return {hash(tuple(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def as_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def jev_ngrams(folder):
    grams = set()
    for tier in ("original", "easy", "hard"):
        for line in open(Path(folder) / f"{tier}.jsonl", encoding="utf-8"):
            grams |= ngrams(as_text(json.loads(line)["state"]))
    return grams


def excluded_tasks():
    meta = yaml.safe_load(open(hf_hub_download(TS, "sources.yaml", repo_type="dataset")))["sources"]
    return {k for k, v in meta.items() if v.get("dataset") in EXCLUDE_DATASETS}


def ts_row(r):
    """A tasksource row in training layout, or None when it is malformed."""
    target = [float(x) for x in r["target"]]
    if r["kind"] == "noul":
        if len(target) != 1 or not 0 <= target[0] <= 1:
            return None
        options, target = ["No", "Yes"], [1 - target[0], target[0]]
    else:
        options = [str(o) for o in r["options"]]
        if len(options) < 2 or len(options) != len(target) or sum(target) <= 0:
            return None
    return {"src": "ts/" + r["source"], "kind": r["kind"], "question": r["question"], "state": r["state"],
            "options": options, "target": target}


def tasksource(out, jev, stats):
    skip = excluded_tasks()
    print(f"excluding {len(skip)} tasks built from proprietary model outputs: {sorted(skip)}", flush=True)
    files = sorted(f for f in list_repo_files(TS, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet") and "/test-" not in f)
    for name in files:
        split = "val" if "/validation-" in name else "train"
        pf = pq.ParquetFile(hf_hub_download(TS, name, repo_type="dataset"))
        for batch in pf.iter_batches(columns=TS_COLS, batch_size=20000):
            for r in batch.to_pylist():
                src = r["source"]
                if src.startswith(EXCLUDE_PREFIXES) or src in skip:
                    stats["ts_excluded"] += 1
                    continue
                if jev and ngrams(r["state"]) & jev:
                    stats["ts_jev_overlap"] += 1
                    continue
                row = ts_row(r)
                if row is None:
                    stats["ts_malformed"] += 1
                    continue
                out[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[f"ts_{split}"] += 1
                stats[f"{split}:{row['src']}"] += 1
                stats[f"kind_{row['kind']}"] += 1
                stats["soft"] += int(max(row["target"]) < 0.999)
        print(f"  {name}: {totals(stats)}", flush=True)


def totals(stats):
    return {k: v for k, v in stats.items() if ":" not in k}


def clean(text):
    if not text or str(text).startswith("empty-name-"):
        return ""
    return re.sub(r"\s+", " ", str(text).replace("_", " ")).strip()[:60]


def label(node):
    value = node.get("value")
    if node.get("role") not in VALUE_ROLES or not isinstance(value, str) or value.lstrip().startswith(("{", "[")):
        value = ""
    return clean(node.get("name")) or clean(node.get("description")) or clean(value)


def parse(text):
    """GUIrilla stores trees as JSON but, in the train split, clicked elements as Python dict reprs."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def flatten(node, depth=0, out=None):
    out = [] if out is None else out
    out.append((depth, node))
    for child in node.get("children") or []:
        flatten(child, depth + 1, out)
    return out


def element_text(node):
    return f'{node.get("role_description") or node.get("role", "").replace("AX", "").lower()} "{label(node)}"'


def outline(nodes, limit=80):
    lines = []
    for depth, n in nodes:
        lab = label(n)
        if lab and n.get("role") != "AXWindow":
            lines.append("  " * min(depth, 6) + element_text(n))
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def gui_rows(r, rng, extra=2):
    """The crawler's own left click plus up to `extra` other labelled controls of the same window,
    each as its own step (the instruction comes from the element either way)."""
    if not str(r["action"]).startswith("left click"):
        return []
    tree, gold = parse(r["accessibility"]), parse(r["element_data"])
    nodes = flatten(tree)
    by_id = {n.get("id"): n for _, n in nodes}
    others = [n for _, n in nodes if n.get("role") in ACTIONABLE and n.get("enabled", True) and label(n)]
    picks = [(by_id.get(gold.get("id")), "gui/guirilla-macos")]
    picks += [(n, "gui/guirilla-macos-extra") for n in rng.sample(others, min(extra, len(others)))]
    out, used = [], set()
    for node, src in picks:
        if node is None or id(node) in used:
            continue
        used.add(id(node))
        row = gui_row(r, tree, nodes, node, src, rng)
        if row:
            out.append(row)
    return out


def gui_row(r, tree, nodes, target, src, rng, k=12):
    if not label(target):
        return None
    gold_text = element_text(target)
    pool, seen = [], {gold_text}
    for _, n in nodes:
        if n is target or n.get("role") not in ACTIONABLE or not label(n):
            continue
        t = element_text(n)
        if t == gold_text and n.get("enabled", True):
            return None
        if t not in seen:
            seen.add(t)
            pool.append(t)
    if not pool:
        return None
    options = rng.sample(pool, min(len(pool), k - 1)) + [gold_text]
    rng.shuffle(options)
    rd = target.get("role_description") or "element"
    templates = TEMPLATES.get(rd, FALLBACK)
    lab = label(target)
    instruction = rng.choice(templates).format(L=lab if rng.random() < 0.8 else lab.lower(), R=rd)
    app = re.sub(r"^com\.|\.", " ", str(r["app_name"])).strip()
    state = f"App: {app}\nWindow: {label(tree) or 'untitled'}\n{outline(nodes)}"
    return {"src": src, "kind": "choice", "question": instruction, "state": state,
            "options": options, "target": [float(o == gold_text) for o in options]}


def gui_shard(name, tmp):
    """One GUIrilla parquet shard (mostly screenshots) downloaded whole, read for its four text
    columns and deleted, so at most a few shards sit on disk at once."""
    made, skipped, errors = [], 0, 0
    path = Path(hf_hub_download(GUI, name, repo_type="dataset", local_dir=tmp))
    try:
        for batch in pq.ParquetFile(path).iter_batches(columns=GUI_COLS, batch_size=256):
            for r in batch.to_pylist():
                seed = int(hashlib.md5((r["accessibility"][:200] + r["element_data"]).encode()).hexdigest()[:8], 16)
                try:
                    rows = gui_rows(r, random.Random(seed))
                except (ValueError, KeyError, TypeError, SyntaxError):
                    errors += 1
                    continue
                skipped += not rows
                made += rows
    finally:
        path.unlink(missing_ok=True)
    return name, made, skipped, errors


def guirilla(out, stats, shards, threads=4):
    files = sorted(f for f in list_repo_files(GUI, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))
    if shards:
        files = [f for f in files if "/test-" in f][:1] + [f for f in files if "/train-" in f][:shards]
    tmp = tempfile.mkdtemp(prefix="guirilla-")
    with ThreadPoolExecutor(threads) as pool:
        for name, made, skipped, errors in pool.map(lambda name: gui_shard(name, tmp), files):
            split = "val" if "/test-" in name else "train"
            for row in made:
                out[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[f"gui_{split}"] += 1
                stats[f"{split}:{row['src']}"] += 1
            stats["gui_skipped"] += skipped
            stats["gui_errors"] += errors
            print(f"  {name.split('/')[-1]}: gui train {stats['gui_train']}, val {stats['gui_val']}, "
                  f"skipped {stats['gui_skipped']}, errors {stats['gui_errors']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev", type=Path, required=True, help="folder with JevBench original/easy/hard.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gui-shards", type=int, default=0, help="quick test: only this many GUIrilla train shards")
    ap.add_argument("--skip-ts", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    jev = jev_ngrams(a.jev)
    print(f"JevBench: {len(jev):,} distinct {NGRAM}-grams", flush=True)
    out = {s: gzip.open(a.out / f"{s}.jsonl.gz", "wt", encoding="utf-8") for s in ("train", "val")}
    guirilla(out, stats, a.gui_shards)
    if not a.skip_ts:
        tasksource(out, jev, stats)
    for f in out.values():
        f.close()
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("done:", json.dumps(totals(stats)), flush=True)


if __name__ == "__main__":
    main()
