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

v1.1 adds three rewrites of the same rows, all built by rule from their human labels:
- yn/...: choice rows asked as yes/no with the criteria spelled out, the way Jev frames noul
  decisions. v1 saw yes/no almost only as UNLI and synthetic
  puzzles and answered "yes" to 62% of JevBench yes/no items. Half the rewrites ask about the
  right option and half about a wrong one, and some flip the polarity ("Should X be rejected?").
- trap/...: banking77 and clinc intents with a second intent mentioned only to rule it out ("I'm
  not asking about card arrival."), as a choice with a "none of these" option and as yes/no.
- gui/...-neg and gui/...-none: steps that name an element not to click ("Don't click Save."),
  and steps whose element is not on screen, answered with "none of these".

v1.2 shows every yes/no decision as two options from model.decider.yes_no, "No: <criteria>" and
"Yes: <question or claim> <criteria>". v1.1 put the criteria in the question and learned nothing
from them (val loss stayed at ln 2): the model reads the options, not the question.

    python data/prep_finetune.py --jev jevbench-public --out prep
"""

import argparse
import ast
import gzip
import hashlib
import json
import random
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.decider import yes_no  # noqa: E402

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
NONE = "none of these"
YES_WORDS, NO_WORDS = {"yes", "true", "correct"}, {"no", "false", "incorrect"}
UNSURE = {"don't know", "dont know", "irrelevant", "clarification needed", "unknown", "not mentioned", "neither"}
YN_RATE, YN_MAX = 0.3, {"train": 6000, "val": 200}
INTENTS = {"ts/banking77", "ts/clinc_oos/plus"}
OUT_OF_SCOPE = {"oos"}
RULED_OUT = ["I'm not asking about {b}.", "This is not about {b}.", "{B} is fine, no help needed there.",
             "Nothing about {b}, by the way.", "Not a {b} question.", "Leave {b} out of it."]
INTENT_QUESTIONS = ["What does the customer actually ask for? Ignore anything they rule out.",
                    "Which request does this message make? A topic that is only mentioned or ruled out is not a request.",
                    "Pick the intent of this message.", "What is this message asking for?"]
GUI_NEG = ["Don't click {B}. {I}", "Not {B}. {I}", "{I}, not {B}", "Leave {B} alone and {i}"]


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
        options, target = yes_no(r["question"]), [1 - target[0], target[0]]
    else:
        options = [str(o) for o in r["options"]]
        if len(options) < 2 or len(options) != len(target) or sum(target) <= 0:
            return None
    return {"src": "ts/" + r["source"], "kind": r["kind"], "question": r["question"], "state": r["state"],
            "options": options, "target": target}


def noul(src, question, state, yes, true, false, claim=None):
    return {"src": src, "kind": "noul", "question": question, "state": state,
            "options": yes_no(question if claim is None else claim, true, false), "target": [1 - yes, yes]}


def yn_row(row, rng):
    """A choice row asked as a yes/no decision with explicit criteria, or None when it does not
    split cleanly into yes and no."""
    opts, t = row["options"], row["target"]
    total, low = sum(t), [o.strip().lower() for o in opts]
    src, q, state = "yn/" + row["src"][3:], row["question"], row["state"]
    yes_i = [i for i, o in enumerate(low) if o in YES_WORDS]
    no_i = [i for i, o in enumerate(low) if o in NO_WORDS]
    if yes_i or no_i:
        if len(yes_i) != 1 or len(no_i) != 1 or not all(o in YES_WORDS | NO_WORDS | UNSURE for o in low):
            return None
        return noul(src, q, state, t[yes_i[0]] / total, "the answer is yes, and the text or the question establishes it.",
                    "the answer is no, or it is not established.")
    best = max(range(len(t)), key=t.__getitem__)
    x = best if rng.random() < 0.5 else rng.choice([i for i in range(len(t)) if i != best])
    name, p = opts[x][:300], t[x] / total
    if 2 < len(opts) <= 6 and all(len(o) <= 80 for o in opts):
        state = f"{state}\n\nOptions: " + " | ".join(opts)
    if rng.random() < 0.7:
        return noul(src, f"{q} Is “{name}” the right answer?", state, p, f"“{name}” is the right answer.",
                    "another answer is right.", "")
    return noul(src, f"{q} Should “{name}” be rejected?", state, 1 - p, f"“{name}” is not the right answer.",
                f"“{name}” is the right answer.", "")


def human(label):
    return label.replace("_", " ").strip()


def intent_traps(utterances, rng):
    """Intent rows rewritten so a second intent is mentioned only to be ruled out: one choice
    (four intents and "none of these") and one yes/no per utterance."""
    rows = []
    labels = sorted({g for _, g in utterances if g not in OUT_OF_SCOPE and g.lower() not in YES_WORDS | NO_WORDS | {"maybe"}})
    for text, gold in utterances:
        if gold not in OUT_OF_SCOPE and gold not in labels:
            continue
        others = rng.sample([lab for lab in labels if lab != gold], 4)
        ruled = others[0]
        b = human(ruled)
        state = text
        if rng.random() < 0.75:
            neg = rng.choice(RULED_OUT).format(b=b, B=b[:1].upper() + b[1:])
            state = f"{neg} {text}" if rng.random() < 0.3 else f"{text} {neg}"
        none = gold in OUT_OF_SCOPE
        options = ([] if none else [gold]) + others[:4 if none else 3]
        rng.shuffle(options)
        names = [human(o) for o in options] + [NONE]
        target = [float(o == gold) for o in options] + [float(none)]
        rows.append({"src": "trap/intent-choice", "kind": "choice", "question": rng.choice(INTENT_QUESTIONS),
                     "state": state, "options": names, "target": target})
        asked = gold if not none and rng.random() < 0.5 else (ruled if state != text and rng.random() < 0.6 else others[1])
        rows.append(noul("trap/intent-noul", f"Does this message ask about {human(asked)}?", state, float(asked == gold),
                         "the message requests this.", "the message does not request it, or only mentions or rules it out."))
    return rows


def tasksource(out, jev, stats):
    skip = excluded_tasks()
    print(f"excluding {len(skip)} tasks built from proprietary model outputs: {sorted(skip)}", flush=True)
    files = sorted(f for f in list_repo_files(TS, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet") and "/test-" not in f)
    rng = random.Random(11)
    utterances = {"train": [], "val": []}

    def write(split, row):
        out[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[f"{split}:{row['src']}"] += 1
        stats[f"kind_{row['kind']}"] += 1

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
                write(split, row)
                stats[f"ts_{split}"] += 1
                stats["soft"] += int(max(row["target"]) < 0.999)
                if row["src"] in INTENTS and max(row["target"]) > 0.999:
                    utterances[split].append((row["state"], row["options"][row["target"].index(max(row["target"]))]))
                if row["kind"] == "choice" and rng.random() < YN_RATE and stats[f"{split}:yn/{row['src'][3:]}"] < YN_MAX[split]:
                    yn = yn_row(row, rng)
                    if yn:
                        write(split, yn)
                        stats[f"yn_{split}"] += 1
        print(f"  {name}: {totals(stats)}", flush=True)
    for split, utts in utterances.items():
        for row in intent_traps(utts, rng):
            write(split, row)
            stats[f"trap_{split}"] += 1
    print(f"  traps: {totals(stats)}", flush=True)


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
        out += gui_row(r, tree, nodes, node, src, rng)
    return out


def gui_row(r, tree, nodes, target, src, rng, k=12):
    """The step itself, plus sometimes a variant that rules out another element by name, or one
    where the element is missing from the window and the answer is "none of these"."""
    if not label(target):
        return []
    gold_text = element_text(target)
    pool, seen = [], {gold_text}
    for _, n in nodes:
        if n is target or n.get("role") not in ACTIONABLE or not label(n):
            continue
        t = element_text(n)
        if t == gold_text and n.get("enabled", True):
            return []
        if t not in seen:
            seen.add(t)
            pool.append(t)
    if not pool:
        return []
    options = rng.sample(pool, min(len(pool), k - 1)) + [gold_text]
    rng.shuffle(options)
    rd = target.get("role_description") or "element"
    templates = TEMPLATES.get(rd, FALLBACK)
    lab = label(target)
    instruction = rng.choice(templates).format(L=lab if rng.random() < 0.8 else lab.lower(), R=rd)
    app = re.sub(r"^com\.|\.", " ", str(r["app_name"])).strip()
    head = f"App: {app}\nWindow: {label(tree) or 'untitled'}\n"
    rows = [{"src": src, "kind": "choice", "question": instruction, "state": head + outline(nodes),
             "options": options, "target": [float(o == gold_text) for o in options]}]
    roll = rng.random()
    if roll < 0.12 and len(options) > 1:
        other = rng.choice([o for o in options if o != gold_text])
        b = other.split('"', 1)[1].rstrip('"')
        q = rng.choice(GUI_NEG).format(B=b, I=instruction, i=instruction[:1].lower() + instruction[1:])
        rows.append({**rows[0], "src": src + "-neg", "question": q})
    elif roll < 0.2 and len(options) > 2:
        rest = [o for o in options if o != gold_text]
        rows.append({"src": src + "-none", "kind": "choice", "question": instruction,
                     "state": head + outline([(d, n) for d, n in nodes if n is not target]),
                     "options": rest + [NONE], "target": [0.0] * len(rest) + [1.0]})
    return rows


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
