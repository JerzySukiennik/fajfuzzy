"""Fine-tune Fajfuzzy into a decision model: question + state + options -> a probability per option.

Training data is one JSON object per line (train.jsonl.gz and val.jsonl.gz in --data, written
by data/prep_finetune.py), in the tasksource-jev layout with yes/no already turned into options:
    {"src": "ts/chaos-mnli-ambiguity/votes", "kind": "choice", "question": "...", "state": "...",
     "options": ["entailment", "neutral", "contradiction"], "target": [0.62, 0.30, 0.08]}
Targets are distributions (human vote shares where a dataset has them, one-hot otherwise) and
the loss is cross-entropy against them, so the model learns how opinions split, not just the
majority. Every epoch takes at most --cap examples per source, so a big dataset cannot drown
the small ones. JevBench (--jev) is only ever scored, never trained on or used to pick a
checkpoint; the best checkpoint is the one with the best mean accuracy across val sources.

    python train/finetune.py --init disc.pt --tokenizer tokenizer.json --data prep --jev jevbench --out run

--init takes the pretrained disc.pt, or an earlier decider.pt to keep fine-tuning it on new data.
"""

import argparse
import gzip
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.decider import OPT, PAD, Decider, encode, soft_loss  # noqa: E402

_tok = None


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def rows(path):
    return (json.loads(line) for line in lines(path))


def jev_row(r, tier):
    """A JevBench item in the training layout: yes/no criteria folded into the question, choice
    and scale options shown by their criteria text, the way tasksource presents them."""
    q, labels = r["question"], r["labels"]
    crit = q.get("criteria")
    state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    if q["type"] == "noul":
        text = q["instructions"]
        if crit:
            text += f" Yes means: {crit.get('true')} No means: {crit.get('false')}"
        return {"src": f"jev/{tier}/noul", "kind": "noul", "question": text, "state": state,
                "options": ["No", "Yes"], "target": [float(r["expected"] != "yes"), float(r["expected"] == "yes")]}
    if isinstance(crit, dict):
        options = [crit.get(label) or label for label in labels]
    elif isinstance(crit, list) and len(crit) == len(labels):
        options = crit
    else:
        options = labels
    return {"src": f"jev/{tier}/{q['type']}", "kind": q["type"], "question": q["instructions"], "state": state,
            "options": options, "target": [float(str(label) == str(r["expected"])) for label in labels]}


def jevbench(folder):
    return [jev_row(r, tier) for tier in ("original", "easy", "hard") for r in rows(Path(folder) / f"{tier}.jsonl")]


def load_init(path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return Decider.from_disc(path) if "rtd_dense.weight" in ck["weights"] else Decider.from_file(path)


def _load_tokenizer(path):
    global _tok
    _tok = Tokenizer.from_file(path)


def _encode_chunk(job):
    chunk, max_len = job
    out = []
    for r in chunk:
        r = json.loads(r) if isinstance(r, str) else r
        try:
            ids = encode(_tok, r["question"], r["state"], r["options"], None, max_len)
        except ValueError:
            continue
        t = np.asarray(r["target"], dtype=np.float32)
        if len(t) != ids.count(OPT) or not t.sum() > 0:
            continue
        out.append((np.asarray(ids, dtype=np.uint16), t / t.sum(), r["src"]))
    return out


def chunks(source, n=2000):
    part = []
    for r in source:
        part.append(r)
        if len(part) == n:
            yield part
            part = []
    if part:
        yield part


def tokenize(tok_path, source, max_len, limit=0, workers=0):
    """(ids as uint16, target, src) per usable row, encoded on every core; at most `limit` rows
    per source when set."""
    kept, seen = [], defaultdict(int)
    with Pool(workers or os.cpu_count() or 1, _load_tokenizer, (str(tok_path),)) as pool:
        for part in pool.imap(_encode_chunk, ((c, max_len) for c in chunks(source))):
            for item in part:
                if limit and seen[item[2]] >= limit:
                    continue
                seen[item[2]] += 1
                kept.append(item)
    return kept


def batches(items, budget, shuffle):
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    out, cur, longest = [], [], 0
    for i in order:
        n = len(items[i][0])
        if cur and max(longest, n) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        random.shuffle(out)
    return out


def collate(items, idx, device):
    rows = [items[i] for i in idx]
    T = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), T), PAD, dtype=torch.long)
    tgt = torch.zeros((len(rows), T))
    for k, (x, t, _) in enumerate(rows):
        ids[k, :len(x)] = torch.from_numpy(x.astype(np.int64))
        tgt[k, :len(x)][ids[k, :len(x)] == OPT] = torch.from_numpy(t)
    lengths = torch.tensor([len(r[0]) for r in rows])
    return ids.to(device), lengths.to(device), tgt.to(device)


@torch.no_grad()
def evaluate(model, items, budget, device, amp):
    model.eval()
    per = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for idx in batches(items, budget, False):
        ids, lengths, tgt = collate(items, idx, device)
        with amp:
            logits = model(ids, lengths)
        p = torch.softmax(logits.float(), -1)
        for k, i in enumerate(idx):
            src = items[i][2]
            m = ids[k] == OPT
            pk, tk = p[k][m], tgt[k][m]
            s = per[src]
            s[0] += int(pk.argmax() == tk.argmax())
            s[1] += 1
            s[2] += float(-(tk * torch.log(pk.clamp_min(1e-9))).sum())
            s[3] += float((pk - tk).abs().sum() / 2)
    model.train()
    return {src: {"acc": a / n, "ce": ce / n, "tv": tv / n, "n": n} for src, (a, n, ce, tv) in sorted(per.items())}


def param_groups(model, lr, head_lr, decay, wd):
    core = model.module if hasattr(model, "module") else model
    L = len(core.disc.blocks)
    groups = []

    def add(params, rate):
        params = list(params)
        groups.append({"params": [p for p in params if p.ndim >= 2], "lr": rate, "weight_decay": wd})
        groups.append({"params": [p for p in params if p.ndim < 2], "lr": rate, "weight_decay": 0.0})

    add(list(core.dense.parameters()) + list(core.out.parameters()), head_lr)
    add(core.disc.ln_f.parameters(), lr)
    for i, block in enumerate(core.disc.blocks):
        add(block.parameters(), lr * decay ** (L - i))
    groups.append({"params": [core.tok.weight], "lr": lr * decay ** (L + 1), "weight_decay": 0.0})
    groups.append({"params": list(core.emb_ln.parameters()), "lr": lr * decay ** (L + 1), "weight_decay": 0.0})
    for g in groups:
        g["base_lr"] = g["lr"]
    return [g for g in groups if g["params"]]


def export(path, model, step, meta):
    core = model.module if hasattr(model, "module") else model
    weights = {k: v.half() for k, v in core.state_dict().items()}
    tmp = Path(path).with_suffix(".tmp")
    torch.save({"weights": weights, "config": asdict(core.cfg), "step": step, "meta": meta}, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="disc.pt from pretraining or a decider.pt")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--jev", type=Path, help="folder with JevBench original/easy/hard.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--jev-len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--cap", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--layer-decay", type=float, default=0.85)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--limit", type=int, default=0, help="quick test: keep only this many rows per source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="tokenizer processes, 0 = every core")
    a = ap.parse_args()

    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    train = defaultdict(list)
    for item in tokenize(a.tokenizer, lines(a.data / "train.jsonl.gz"), a.max_len, a.limit, a.workers):
        train[item[2]].append(item)
    val = tokenize(a.tokenizer, lines(a.data / "val.jsonl.gz"), a.max_len, a.limit, a.workers)
    jev = tokenize(a.tokenizer, jevbench(a.jev), a.jev_len, 0, a.workers) if a.jev else []
    for s, items in sorted(train.items()):
        print(f"train {s}: {len(items):,} (epoch takes {min(len(items), a.cap):,})", flush=True)
    print(f"val {len(val):,}, jevbench {len(jev)}, tokenized in {time.time() - t0:.0f} s", flush=True)

    model = load_init(a.init).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.head_lr, a.layer_decay, a.weight_decay), betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    total = len(batches([x for v in train.values() for x in v[:a.cap]], a.batch_tokens, False)) * a.epochs
    warm = max(1, int(total * a.warmup))
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "log.jsonl", "a")
    best, step = -1.0, 0

    def check(final=False):
        nonlocal best
        v = evaluate(model, val, a.batch_tokens, device, amp)
        score = sum(x["acc"] for x in v.values()) / max(1, len(v))
        rec = {"step": step, "val_mean_acc": round(score, 4), "val": v}
        if jev:
            j = evaluate(model, jev, a.batch_tokens // 2, device, amp)
            hits = sum(x["acc"] * x["n"] for x in j.values())
            rec["jev"] = j
            rec["jev_acc"] = round(hits / sum(x["n"] for x in j.values()), 4)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        jtxt = (f"  jevbench {rec['jev_acc']:.3f} (" + ", ".join(f"{k[4:]} {x['acc']:.2f}" for k, x in rec["jev"].items()) + ")") if jev else ""
        gui = "  ".join(f"{s} {x['acc']:.3f}" for s, x in v.items() if s.startswith("gui/"))
        print(f"  -> step {step}: val mean acc {score:.4f} over {len(v)} sources{jtxt}", flush=True)
        if gui:
            print(f"     {gui}", flush=True)
        if score > best:
            best = score
            export(a.out / "decider.pt", model, step, {"val_mean_acc": score, "jev_acc": rec.get("jev_acc")})
        if final:
            export(a.out / "decider-last.pt", model, step, {"val_mean_acc": score, "jev_acc": rec.get("jev_acc")})

    t_eval = time.time()
    check()
    eval_s, t_train = time.time() - t_eval, time.time()
    stop = False
    for epoch in range(a.epochs):
        items = [x for v in train.values() for x in random.sample(v, min(len(v), a.cap))]
        tl, tn, tt = 0.0, 0, time.time()
        for idx in batches(items, a.batch_tokens, True):
            ids, lengths, tgt = collate(items, idx, device)
            lr_scale = min(1.0, (step + 1) / warm) * max(0.0, 1 - max(0, step - warm) / max(1, total - warm))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * max(lr_scale, 0.02)
            with amp:
                loss = soft_loss(model(ids, lengths), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            tl, tn = tl + float(loss), tn + 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {tl / tn:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{tn / (time.time() - tt):.1f} it/s", flush=True)
                tl, tn, tt = 0.0, 0, time.time()
            if a.max_hours and step == 300:
                per_step = (time.time() - t_train) / step
                left = a.max_hours * 3600 - (time.time() - t0) - 2 * eval_s
                fit = step + int(left / (per_step + eval_s / a.eval_every))
                if fit < total:
                    print(f"{a.max_hours} h fit {fit:,} of {total:,} steps, schedule shortened", flush=True)
                    total = fit
            if step % a.eval_every == 0:
                check()
            if step >= total or (a.max_hours and time.time() - t0 > a.max_hours * 3600):
                stop = True
                break
        if stop:
            break
    check(final=True)
    print(f"done at step {step} in {(time.time() - t0) / 60:.0f} min; best val mean acc {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
