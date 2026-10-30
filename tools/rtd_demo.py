"""Fajfuzzy demo page: the pretrained discriminator spotting swapped words.

Everything is scored on real validation text in full windows, the way the model trained:
words swapped by its own generator (the training task) and common human mix-ups
(three/tree, would/wood, then/than...) injected into real paragraphs, one per paragraph.
Writes one self-contained HTML page in Polish.

    python tools/rtd_demo.py --ckpt s1/run1/ckpt.pt --prep prep --docs 2080,2108 --out page.html
"""

import argparse
import html
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.electra import Config, Electra  # noqa: E402

PAIRS = [(" three", " tree"), (" would", " wood"), (" then", " than"), (" than", " then"), (" their", " there"),
         (" there", " their"), (" buy", " by"), (" off", " of"), (" whether", " weather"), (" accept", " except"),
         (" too", " to"), (" your", " you"), (" quite", " quiet"), (" were", " where"), (" launched", " lunched"),
         (" lose", " loose"), (" its", " it's"), (" affect", " effect")]
WORD = re.compile(r"[A-Za-z0-9'’]+|[^\sA-Za-z0-9]")
SENTENCE = re.compile(r"[^\n.!?]+(?:[.!?]+|$)")


def byte_map():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def pct(p, d=0):
    return f"{p * 100:.{d}f}".replace(".", ",") + "%"


class Model:
    def __init__(self, ckpt, tok_path, threads):
        torch.set_num_threads(threads)
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False, mmap=True)
        self.cfg = Config(**ck["config"])
        self.net = Electra(self.cfg).eval()
        self.net.load_state_dict(ck["model"])
        self.step, self.tokens = ck["step"], ck.get("tokens", 0)
        self.tok = Tokenizer.from_file(str(tok_path))
        bm = byte_map()
        self.bytes = [bytes(bm[c] for c in self.tok.id_to_token(i)) for i in range(self.tok.get_vocab_size())]

    @torch.no_grad()
    def rtd(self, ids):
        m = self.net
        h = m.disc(m.emb_ln(F.embedding(ids, m.tok.weight) + m.delta(ids)))
        return torch.sigmoid(m.rtd_out(F.gelu(m.rtd_dense(h), approximate="tanh")).squeeze(-1))

    @torch.no_grad()
    def corrupt(self, ids, gen):
        m, cfg = self.net, self.cfg
        sel = (torch.rand(ids.shape, generator=gen) < cfg.mask_rate) & (ids >= cfg.n_special)
        hidden = m.gen(m.gen_in(m.gen_emb_ln(m.tok(ids.masked_fill(sel, cfg.mask_id)))))[sel]
        logits = F.linear(m.gen_ln(F.gelu(m.gen_dense(hidden), approximate="tanh")), m.tok.weight, m.gen_bias)
        u = torch.rand(logits.shape, generator=gen).clamp_(1e-10, 1.0 - 1e-7)
        return ids.masked_scatter(sel, (logits - torch.log(-torch.log(u))).argmax(-1))

    def window(self, ids, whole, size=256):
        return torch.tensor([[self.cfg.cls_id] + ids[:size] + ([self.cfg.sep_id] if whole and len(ids) <= size else [])])

    def scores(self, ids, whole):
        return self.rtd(self.window(ids, whole))[0, 1:1 + min(len(ids), 256)].tolist()

    def spans(self, ids):
        buf, out = b"", []
        for i in ids:
            start = len(buf.decode("utf-8", "ignore"))
            buf += self.bytes[i]
            out.append((start, len(buf.decode("utf-8", "ignore"))))
        return buf.decode("utf-8", "ignore"), out

    def words(self, ids, probs, orig=None):
        text, sp = self.spans(ids)
        res = []
        for m in WORD.finditer(text):
            idx = [k for k, (s, e) in enumerate(sp) if s < m.end() and e > m.start()]
            w = {"start": m.start(), "end": m.end(), "text": m.group(0), "p": max(probs[k] for k in idx)}
            if orig is not None and any(orig[k] != ids[k] for k in idx):
                w["was"] = self.spans(orig[idx[0]:idx[-1] + 1])[0].strip()
                w["now"] = self.spans(ids[idx[0]:idx[-1] + 1])[0].strip()
            res.append(w)
        return text, res

    def stats(self, val, n, block, batch, seed):
        rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
        ps, rs = [], []
        for b in range(0, n, batch):
            starts = rng.integers(0, len(val) - block, size=min(batch, n - b))
            x = torch.tensor(np.stack([val[s:s + block] for s in starts]).astype(np.int64).tolist())
            x[:, 0] = self.cfg.cls_id
            c = self.corrupt(x, gen)
            ok = x >= self.cfg.n_special
            ps.append(self.rtd(c)[ok].numpy())
            rs.append(((c != x) & ok)[ok].numpy())
        p, r = np.concatenate(ps), np.concatenate(rs)
        ranks = np.empty(len(p))
        ranks[p.argsort()] = np.arange(1, len(p) + 1)
        pos, neg = int(r.sum()), int((~r).sum())
        flag = p > 0.5
        return {"recall": (flag & r).sum() / pos, "precision": (flag & r).sum() / max(1, flag.sum()),
                "false_alarm": (flag & ~r).sum() / neg, "auc": (ranks[r].sum() - pos * (pos + 1) / 2) / (pos * neg),
                "base": pos / len(p), "windows": n, "block": block}

    def mixups(self, docs, per_type):
        found = {a: [] for a, _ in PAIRS}
        for k, d in enumerate(docs):
            if len(d) < 150:
                continue
            ids, whole = [int(x) for x in d[:256]], len(d) <= 256
            text = self.spans(ids)[0]
            if not text.isascii():
                continue
            for a, b in PAIRS:
                if len(found[a]) >= per_type:
                    continue
                hit = re.search(re.escape(a) + r"\b", text[60:])
                if not hit:
                    continue
                s = 60 + hit.start()
                bad = self.tok.encode(text[:s] + b + text[s + len(a):]).ids
                tb, wb = self.words(bad[:256], self.scores(bad, whole))
                _, wc = self.words(ids, self.scores(ids, whole))
                tgt = next((w for w in wb if w["start"] == s + 1), None)
                same = next((w for w in wc if w["start"] == s + 1), None)
                if tgt is None or same is None:
                    continue
                found[a].append({"doc": k, "pair": (a.strip(), b.strip()), "text": tb, "words": wb, "target": tgt,
                                 "rank": 1 + sum(w["p"] > tgt["p"] for w in wb), "clean": same["p"]})
                break
            if all(len(v) >= per_type for v in found.values()):
                break
        return found

    def paragraph(self, doc, seed, show=110):
        ids = [int(t) for t in doc[:256]]
        c = self.corrupt(self.window(ids, len(doc) <= 256), torch.Generator().manual_seed(seed))[0]
        body = c[1:1 + len(ids)].tolist()
        p = self.rtd(c.unsqueeze(0))[0, 1:1 + len(ids)].tolist()
        text, ws = self.words(body, p, orig=ids)
        cut = self.spans(body[:show])[0]
        end = max(cut.rfind(". "), cut.rfind(".\n")) + 1 or len(cut)
        return text[:end], [w for w in ws if w["end"] <= end]


def documents(val, sep):
    cuts = np.flatnonzero(val == sep)
    return [val[a:b] for a, b in zip(np.concatenate([[0], cuts[:-1] + 1]), cuts)]


def paint(text, ws, offset=0):
    out, pos = [], 0
    for w in ws:
        a, b = w["start"] - offset, w["end"] - offset
        out.append(html.escape(text[pos:a]))
        style = f' style="background:rgba(255,184,0,{min(1.0, w["p"]) * 0.85:.2f})"' if w["p"] >= 0.12 else ""
        tip = f"podejrzenie {pct(w['p'])}" + (f" · było: {w['was']}" if "was" in w else "")
        cls = "w sw" if "was" in w else "w"
        out.append(f'<span class="{cls}"{style} title="{html.escape(tip)}">{html.escape(text[a:b])}</span>')
        pos = b
    return "".join(out) + html.escape(text[pos:])


def mixup_row(c):
    t, s = c["target"], c["target"]["start"]
    sent = next(m for m in SENTENCE.finditer(c["text"]) if m.start() <= s < m.end())
    a = sent.start() + len(sent.group(0)) - len(sent.group(0).lstrip())
    ws = [w for w in c["words"] if a <= w["start"] and w["end"] <= sent.end()]
    for w in ws:
        if w is t:
            w["was"] = c["pair"][0]
    place = "najbardziej podejrzane" if c["rank"] == 1 else f"{c['rank']}. najbardziej podejrzane"
    meta = (f"„{c['pair'][1]}” zamiast „{c['pair'][0]}”: {pct(t['p'])} · {place} z {len(c['words'])} słów akapitu"
            f" · w poprawnym tekście „{c['pair'][0]}”: {pct(c['clean'])}")
    return (f'<div class="pair"><p class="s">{paint(c["text"][a:sent.end()], ws, a)}</p>'
            f'<p class="meta">{html.escape(meta)}</p></div>')


def mixup_table(found):
    rows = []
    for a, cs in found.items():
        if cs:
            rows.append((sum(c["rank"] == 1 for c in cs), sum(c["target"]["p"] > 0.5 for c in cs), len(cs), cs[0]["pair"]))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    body = "".join(f"<tr><td>{html.escape(p[0])} → <b>{html.escape(p[1])}</b></td><td>{top}/{n}</td><td>{hit}/{n}</td></tr>"
                   for top, hit, n, p in rows)
    return ('<table><thead><tr><th>pomyłka</th><th>najbardziej podejrzane</th><th>ponad 50%</th></tr></thead>'
            f"<tbody>{body}</tbody></table>"), [r[3][0] for r in rows]


def article(m, doc, seed):
    text, ws = m.paragraph(doc, seed)
    items = []
    for w in ws:
        if "was" in w:
            state = pct(w["p"]) if w["p"] > 0.5 else f"{pct(w['p'])}, przeoczone"
            items.append(f"<li>{html.escape(w['was'])} → <b>{html.escape(w['now'])}</b> · {state}</li>")
    for w in ws:
        if "was" not in w and w["p"] > 0.5:
            items.append(f"<li>fałszywy alarm: {html.escape(w['text'])} · {pct(w['p'])}</li>")
    return f'<article><p class="s">{paint(text, ws)}</p><ul class="swaps">{"".join(items)}</ul></article>', text


PAGE = """<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fajfuzzy · podmienione słowa</title>
<style>
:root {{ --ink:#1d1d1f; --mute:#6e6e73; --line:#e5e5ea; --card:#f5f5f7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#fff; color:var(--ink); font:17px/1.55 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; -webkit-font-smoothing:antialiased; }}
main {{ max-width:780px; margin:0 auto; padding:56px 16px 72px; }}
.kicker {{ font:600 12px/1.2 ui-monospace, "SF Mono", Menlo, monospace; letter-spacing:.06em; text-transform:uppercase; color:var(--mute); margin:0 0 14px; }}
h1 {{ font:700 42px/1.06 -apple-system, "SF Pro Display", sans-serif; letter-spacing:-.025em; margin:0 0 16px; }}
.lede {{ font-size:19px; color:var(--mute); margin:0 0 36px; max-width:640px; }}
.stats {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin:0 0 12px; }}
.stat {{ background:var(--card); border-radius:16px; padding:16px 16px 14px; }}
.stat b {{ display:block; font:700 32px/1 -apple-system, "SF Pro Display", sans-serif; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
.stat span {{ display:block; margin-top:8px; font-size:13px; line-height:1.35; color:var(--mute); }}
.note {{ font-size:14px; color:var(--mute); margin:0 0 56px; }}
section {{ margin:0 0 56px; }}
h2 {{ font:700 27px/1.2 -apple-system, "SF Pro Display", sans-serif; letter-spacing:-.015em; margin:0 0 8px; }}
.intro {{ color:var(--mute); margin:0 0 18px; }}
.sum {{ font-size:19px; margin:0 0 18px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px 20px; font-size:14px; color:var(--mute); margin:0 0 18px; }}
.w {{ border-radius:4px; padding:0 1px; margin:0 -1px; }}
.sw {{ text-decoration:underline; text-decoration-thickness:2px; text-underline-offset:4px; }}
table {{ width:100%; border-collapse:collapse; margin:0 0 26px; font-size:15px; font-variant-numeric:tabular-nums; }}
th {{ text-align:left; font-weight:600; font-size:13px; color:var(--mute); padding:0 8px 8px 0; border-bottom:1px solid var(--line); }}
td {{ padding:7px 8px 7px 0; border-bottom:1px solid var(--line); }}
td:first-child {{ font-family:ui-monospace, "SF Mono", Menlo, monospace; font-size:14px; }}
.pair {{ border-top:1px solid var(--line); padding:14px 0 16px; }}
.s {{ margin:0; font-size:18px; }}
.meta {{ margin:8px 0 0; font:13px/1.45 ui-monospace, "SF Mono", Menlo, monospace; color:var(--mute); }}
article {{ background:var(--card); border-radius:16px; padding:20px; margin:0 0 12px; }}
article .s {{ font-size:17px; line-height:1.75; white-space:pre-line; }}
.swaps {{ list-style:none; padding:0; margin:14px 0 0; display:flex; flex-wrap:wrap; gap:4px 16px; font:13px/1.45 ui-monospace, "SF Mono", Menlo, monospace; color:var(--mute); }}
.swaps b {{ color:var(--ink); font-weight:600; }}
footer {{ border-top:1px solid var(--line); padding-top:18px; font-size:14px; color:var(--mute); }}
footer p {{ margin:0 0 8px; }}
@media (max-width:680px) {{ main {{ padding-top:36px; }} h1 {{ font-size:32px; }} .stats {{ grid-template-columns:1fr 1fr; }} .stat b {{ font-size:28px; }} }}
</style>
</head>
<body>
<main>
<p class="kicker">Fajfuzzy · checkpoint z sesji {session}</p>
<h1>Już wyłapuje podmienione słowa</h1>
<p class="lede">Krok {step} z 36 000, czyli {done} treningu. Wszystko poniżej policzył prawdziwy model na tekstach, których nigdy nie widział.</p>

<div class="stats">
<div class="stat"><b>{recall}</b><span>podmian złapanych (alarm powyżej 50%)</span></div>
<div class="stat"><b>{precision}</b><span>alarmów trafionych</span></div>
<div class="stat"><b>{false_alarm}</b><span>zwykłych słów z fałszywym alarmem</span></div>
<div class="stat"><b>{auc}</b><span>szans, że podmiana wypadnie podejrzaniej niż zwykłe słowo</span></div>
</div>
<p class="note">{windows} okien walidacji po {block} tokenów. Generator podmienia słowa tak jak w treningu (ok. {base} słów). Model alarmuje rzadko, ale jak już, to zwykle słusznie.</p>

<section>
<h2>Ludzkie pomyłki</h2>
<p class="intro">Wstawiłem typowe pomyłki w prawdziwe akapity, po jednej na akapit z ok. 220 słów. Tego model nigdy nie ćwiczył: w treningu słowa podmieniał mu generator, nie człowiek.</p>
<p class="sum">{mix_n} pomyłek. W {mix_top} przypadków to najbardziej podejrzane słowo w całym akapicie, w {mix_hit} model daje mu ponad 50%. To samo miejsce w poprawnym tekście dostaje zwykle {mix_clean}.</p>
{mix_table}
<div class="legend"><span><span class="w" style="background:rgba(255,184,0,.8)">żółte tło</span> podejrzenie modelu, im mocniejsze, tym pewniejszy</span><span><span class="w sw">podkreślone</span> wstawiona pomyłka</span></div>
{mix_rows}
</section>

<section>
<h2>Tak wygląda trening</h2>
<p class="intro">Prawdziwe akapity z walidacji. Generator podmienia część słów, Fajfuzzy zgaduje które.</p>
<div class="legend"><span><span class="w sw">podkreślone</span> prawdziwa podmiana</span><span><span class="w" style="background:rgba(255,184,0,.8)">żółte tło</span> podejrzenie modelu</span></div>
{articles}
</section>

<footer>
<p>Na pojedynczych, krótkich zdaniach jeszcze nie działa: w pretrainingu widzi tylko długie okna tekstu. Krótkie wejścia (stan plus pytanie) dostanie w fine-tuningu na decyzje.</p>
<p>Checkpoint z sesji {session}: krok {step}, przeczytane {tokens} z 4,72 mld tokenów. Pretraining kończy się ok. 1.11. Liczone na CPU w {mins} min.</p>
</footer>
</main>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--prep", type=Path, required=True, help="folder with tokenizer.json and decide_val.bin")
    ap.add_argument("--docs", default="", help="validation document indices shown as training paragraphs")
    ap.add_argument("--session", default="1")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--per-type", type=int, default=6)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    t0 = time.time()
    m = Model(a.ckpt, a.prep / "tokenizer.json", a.threads)
    val = np.memmap(a.prep / "decide_val.bin", dtype=np.uint16, mode="r")
    docs = documents(val, m.cfg.sep_id)

    st = m.stats(val, a.windows, 512, 8, seed=7)
    print(f"step {m.step}: recall {st['recall']:.3f} precision {st['precision']:.3f} false {st['false_alarm']:.4f} "
          f"auc {st['auc']:.3f} base {st['base']:.3f} ({time.time() - t0:.0f} s)", flush=True)

    found = m.mixups(docs, a.per_type)
    cases = [c for cs in found.values() for c in cs]
    table, order = mixup_table(found)
    picks = order[:a.examples - 3] + order[-3:]
    rows = [mixup_row(found[" " + w][0]) for w in picks]
    for w in picks:
        c = found[" " + w][0]
        print(f"  {c['pair']}: p {c['target']['p']:.2f} rank {c['rank']}/{len(c['words'])} clean {c['clean']:.2f} doc {c['doc']}")
    print(f"mixups {len(cases)}: top1 {np.mean([c['rank'] == 1 for c in cases]):.2f} "
          f"hit {np.mean([c['target']['p'] > 0.5 for c in cases]):.2f} ({time.time() - t0:.0f} s)", flush=True)

    articles = []
    for k in [int(x) for x in a.docs.split(",") if x]:
        row, text = article(m, docs[k], seed=k)
        articles.append(row)
        print(f"  doc {k}: {text[:100]!r}")

    page = PAGE.format(
        session=a.session, step=f"{m.step:,}".replace(",", " "), done=pct(m.step / 36000),
        recall=pct(st["recall"]), precision=pct(st["precision"]), false_alarm=pct(st["false_alarm"], 1),
        auc=pct(st["auc"]), base=pct(st["base"]), windows=st["windows"], block=st["block"],
        mix_n=len(cases), mix_top=pct(np.mean([c["rank"] == 1 for c in cases])),
        mix_hit=pct(np.mean([c["target"]["p"] > 0.5 for c in cases])),
        mix_clean=pct(float(np.median([c["clean"] for c in cases]))), mix_table=table, mix_rows="\n".join(rows),
        articles="\n".join(articles), tokens=f"{m.tokens / 1e9:.2f}".replace(".", ","),
        mins=max(1, round((time.time() - t0) / 60)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page, encoding="utf-8")
    print(f"wrote {a.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
