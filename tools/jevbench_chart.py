"""Draw Fajfuzzy's JevBench result next to every system on the open-weights board as one tall PNG:
headline numbers, Intelligence on the same 231 items, and the full Capability ranking. Writes the
HTML and renders it with headless Chrome.

    python tools/jevbench_chart.py --bench bench.json --board board-live.json --out jevbench.png
"""

import argparse
import html
import json
import subprocess
from pathlib import Path

from PIL import Image

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH, SCALE = 1200, 2
ENCODER_KEYS = ("gliner",)

CSS = """
:root { color-scheme: light; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ours: #2a78d6; --ours-deep: #1c5cab; --jev: #eb6834; --enc: #1baf7a;
  --other: #cfcdc5; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--surface); color: var(--ink); width: 1200px; padding: 56px 56px 48px;
  font-family: -apple-system, "SF Pro Text", system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
.eyebrow { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
h1 { font-family: -apple-system, "SF Pro Display", system-ui, sans-serif; font-size: 40px; line-height: 1.1;
  letter-spacing: -.02em; font-weight: 700; margin-top: 10px; }
.lede { font-size: 16px; line-height: 1.45; color: var(--ink-2); margin-top: 12px; max-width: 900px; }
.tiles { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-top: 32px; }
.tile { border: 1px solid rgba(11,11,11,.08); border-radius: 14px; padding: 16px 16px 14px; background: #fff; }
.tile.hero { border-color: rgba(42,120,214,.35); background: rgba(42,120,214,.05); }
.tile .label { font-size: 12px; color: var(--ink-2); font-weight: 500; }
.tile .value { font-family: -apple-system, "SF Pro Display", system-ui, sans-serif; font-size: 34px;
  font-weight: 650; letter-spacing: -.02em; margin-top: 6px; }
.tile .note { font-size: 11.5px; line-height: 1.4; color: var(--muted); margin-top: 6px; }
section { margin-top: 48px; }
h2 { font-size: 21px; font-weight: 650; letter-spacing: -.01em; }
.sub { font-size: 13px; line-height: 1.45; color: var(--ink-2); margin-top: 6px; max-width: 980px; }
.legend { display: flex; gap: 20px; margin-top: 14px; font-size: 12px; color: var(--ink-2); flex-wrap: wrap; }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.legend i { width: 14px; height: 8px; border-radius: 0 4px 4px 0; display: inline-block; }
.plot { position: relative; margin-top: 16px; }
.grid { position: absolute; top: 0; bottom: 0; left: 308px; right: 56px; pointer-events: none; }
.grid b { position: absolute; top: 18px; bottom: 0; width: 1px; background: var(--grid); }
.grid b.zero { background: var(--axis); }
.grid em { position: absolute; top: 0; transform: translateX(-50%); font-style: normal; font-size: 10.5px;
  color: var(--muted); font-variant-numeric: tabular-nums; }
.rows { position: relative; padding-top: 22px; }
.row { display: grid; grid-template-columns: 34px 250px 1fr 44px; column-gap: 12px; align-items: center;
  height: 13px; font-size: 10.5px; line-height: 13px; }
.row .rk { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.row .nm { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--ink-2); text-align: right; }
.row .val { font-variant-numeric: tabular-nums; color: var(--ink-2); }
.track { position: relative; height: 13px; }
.bar { position: absolute; left: 0; top: 2.5px; height: 8px; border-radius: 0 4px 4px 0; background: var(--other); }
.row.jev .nm, .row.jev .val { color: var(--ink); font-weight: 600; }
.row.ours { height: 26px; margin: 4px 0; font-size: 12.5px; background: rgba(42,120,214,.10); border-radius: 6px; }
.row.ours .nm, .row.ours .val, .row.ours .rk { color: var(--ink); font-weight: 700; }
.row.ours .track { height: 26px; }
.row.ours .bar { top: 7px; height: 12px; border-radius: 0 6px 6px 0; }
.band { position: absolute; top: 12.5px; height: 1.5px; background: var(--ours-deep); }
.band::before, .band::after { content: ""; position: absolute; top: -4px; width: 1.5px; height: 9.5px; background: var(--ours-deep); }
.band::before { left: 0; } .band::after { right: 0; }
.notes { margin-top: 44px; border-top: 1px solid var(--grid); padding-top: 18px; display: grid;
  grid-template-columns: 1fr 1fr; gap: 10px 32px; }
.notes p { font-size: 12px; line-height: 1.5; color: var(--ink-2); }
.notes b { color: var(--ink); font-weight: 600; }
.foot { margin-top: 18px; font-size: 11px; color: var(--muted); }
"""


def short(title, n=40):
    name = (title or "").split(" (")[0].split(" · ")[0].strip()
    return name if len(name) <= n else name[:n - 1].rstrip() + "…"


def kind(cat, key):
    if key == "fajfuzzy-v1":
        return "ours"
    if cat == "jev-reference" or key.startswith("jev-1."):
        return "jev"
    if cat == "open-encoder" or key.startswith(ENCODER_KEYS):
        return "enc"
    return "other"


def colour(k):
    return {"ours": "var(--ours)", "jev": "var(--jev)", "enc": "var(--enc)", "other": "var(--other)"}[k]


def plot(rows, band=None):
    ticks = "".join(f'<b class="{"zero" if t == 0 else ""}" style="left:{t}%"></b><em style="left:{t}%">{t}</em>'
                    for t in (0, 25, 50, 75, 100))
    out = [f'<div class="plot"><div class="grid">{ticks}</div><div class="rows">']
    for r in rows:
        v = max(0.0, min(100.0, r["value"]))
        extra = ""
        if r["kind"] == "ours" and band:
            extra = f'<div class="band" style="left:{band[0]:.2f}%;width:{band[1] - band[0]:.2f}%"></div>'
        out.append(f'<div class="row {r["kind"]}"><span class="rk">{html.escape(r["rank"])}</span>'
                   f'<span class="nm">{html.escape(r["name"])}</span><span class="track">'
                   f'<span class="bar" style="width:{v:.2f}%;background:{colour(r["kind"])}"></span>{extra}</span>'
                   f'<span class="val">{r["label"]}</span></div>')
    out.append("</div></div>")
    return "".join(out)


def legend():
    items = [("ours", "Fajfuzzy v1"), ("jev", "Jev (wzorzec, zamknięty)"), ("enc", "Enkodery i klasyfikatory, klasa Fajfuzzy"),
             ("other", "Pozostałe: dekodery LLM, dyfuzyjne, rerankery, modele bazowe")]
    return '<div class="legend">' + "".join(f'<span><i style="background:{colour(k)}"></i>{t}</span>' for k, t in items) + "</div>"


def page(bench, board):
    b, pub, lat = bench["board_estimate"], bench["public_items"], bench["latency"]
    jev = next(r for r in board if r["key"] == "jev-1.13.0")
    encoders = sorted([r["capability"] for r in board if r["cat"] == "open-encoder"] + [b["capability"]], reverse=True)
    enc_rank = encoders.index(b["capability"]) + 1
    same_jev = next(r for r in bench["same_items"] if r["key"] == "jev-1.13.0")
    same_rank = [r["key"] for r in bench["same_items"]].index("fajfuzzy-v1") + 1

    full = [{"key": r["key"], "name": short(r["title"]), "value": r["capability"], "label": f'{r["capability"]:.1f}',
             "rank": r["rank"], "kind": kind(r["cat"], r["key"])} for r in board]
    full.append({"key": "fajfuzzy-v1", "name": "Fajfuzzy v1 · 110M · szacunek", "value": b["capability"],
                 "label": f'{b["capability"]:.1f}', "rank": f'≈{b["rank_jev_class"]}', "kind": "ours"})
    full.sort(key=lambda r: -r["value"])
    for r in full:
        if r["rank"].isdigit() and r["value"] < b["capability"]:
            r["rank"] = str(int(r["rank"]) + 1)
    names = [short(r["display"]) for r in bench["same_items"]]
    same = [{"key": r["key"], "name": "Fajfuzzy v1 · 110M" if r["key"] == "fajfuzzy-v1"
             else (r["key"] if names.count(name) > 1 else name),
             "value": r["i_same_items"], "label": f'{r["i_same_items"]:.1f}', "rank": str(i + 1),
             "kind": kind(r["cat"], r["key"])} for i, (r, name) in enumerate(zip(bench["same_items"], names))]

    tiles = [
        ("hero", "Capability Score", f'≈{b["capability"]:.1f}',
         f'≈{b["rank_jev_class"]}. miejsce z {b["n_jev_class"] + 1} w rankingu Jev-class; Jev 1.13.0 ma {jev["capability"]:.1f}'),
        ("", "Intelligence", f'≈{b["intelligence"]:.0f}',
         f'na skali tablicy (przedział {b["intelligence_band"][0]:.0f}-{b["intelligence_band"][1]:.1f}); '
         f'na tych samych pytaniach {pub["intelligence"]:.1f}, Jev {same_jev["i_same_items"]:.1f}'),
        ("", "Calibration", f'{pub["calibration"]:.1f}', f'czy pewność zgadza się z trafnością; Jev {jev["calibration"]:.1f}'),
        ("", "Wśród enkoderów", f'{enc_rank}. z {len(encoders)}', "małe modele typu BERT, ta sama klasa co Fajfuzzy"),
        ("", "Opóźnienie p50", f'{lat["p50_adjusted_s"]:.2f} s',
         f'liczone jak na tablicy: 2 × zmierzone {lat["p50_measured_s"]:.2f} s + 0.15 s; CPU, {lat["threads"]} wątki'),
    ]
    tiles_html = "".join(f'<div class="tile {c}"><div class="label">{l}</div><div class="value">{v}</div>'
                         f'<div class="note">{html.escape(n)}</div></div>' for c, l, v, n in tiles)
    fit = b["fit"]
    notes = [
        f'<b>Pomiar.</b> Fajfuzzy przeszedł przez oficjalny runner i scorer JevBench (repo fstandhartinger/jevbench, '
        f'commit 609a664): {bench["items"]} publicznych pytań, jedno zapytanie na decyzję, Kaggle CPU. JevBench nie był '
        f'używany do treningu ani wyboru checkpointu.',
        '<b>Wzory.</b> Intelligence: trafność ponad zgadywanie, osobno dla choice, noul (tak/nie) i score, ważona poziomami '
        'trudności 0.1/0.2/0.3/0.4, potem średnia. Calibration: ECE i NRPS jak na tablicy. Oba wzory sprawdzone co do '
        'cyfry na 138 systemach z opublikowanego agregatu.',
        f'<b>Szacunek.</b> Tablica liczy 1500 pytań (300 publicznych i 1200 ukrytych), których nie da się pobrać. '
        f'Dlatego Intelligence Fajfuzzy jest przeniesiona na skalę tablicy z {fit["n"]} systemów zmierzonych na tych '
        f'samych {bench["items"]} pytaniach (r = {fit["r"]:.2f}); kreska przy pasku to przedział tego szacunku.',
        f'<b>Gdzie traci.</b> Na łatwych pytaniach wyboru trafia {pub["tiers"]["choice|easy"]["accuracy"] * 100:.0f}%, '
        f'na trudnych {pub["tiers"]["choice|hard"]["accuracy"] * 100:.0f}%, a na tak/nie '
        f'{sum(pub["tiers"][k]["correct"] for k in pub["tiers"] if k.startswith("noul")) / sum(pub["tiers"][k]["n"] for k in pub["tiers"] if k.startswith("noul")) * 100:.0f}%, '
        'czyli gorzej niż rzut monetą. Calibration jest przyzwoita: model wie, że zgaduje.',
    ]
    return f"""<!doctype html><html lang="pl"><head><meta charset="utf-8"><title>Fajfuzzy na JevBench</title>
<style>{CSS}</style></head><body>
<div class="eyebrow">JevBench · Fajfuzzy v1 · 2.11.2026</div>
<h1>Fajfuzzy v1 na tablicy JevBench</h1>
<p class="lede">Capability Score to średnia Intelligence i Calibration. Obok Fajfuzzy wszystkie {len(board)} systemy z
benchmarkheaven.com/jev-models, w tym Jev od TypeSafe jako wzorzec.</p>
<div class="tiles">{tiles_html}</div>
<section><h2>Te same {bench["items"]} pytania, bez szacowania</h2>
<p class="sub">Intelligence na identycznych publicznych pytaniach: 0 to poziom zgadywania, 100 to komplet. Wyniki
pozostałych {len(same) - 1} systemów pochodzą z repo scorera JevBench. Fajfuzzy jest tu {same_rank}. z {len(same)}.</p>
{legend()}{plot(same)}</section>
<section><h2>Cała tablica: Capability Score</h2>
<p class="sub"># to miejsce wśród systemów Jev-class (koszt i opóźnienie najwyżej 2× Jev) po dopisaniu Fajfuzzy; „–” to
wzorzec albo system poza limitem. Fajfuzzy mieści się w obu limitach (koszt ok. ${bench["cost"]["usd_per_1000_estimate"]:.4f} na 1000 decyzji).</p>
{legend()}{plot(full, b["capability_band"])}</section>
<div class="notes">{"".join(f"<p>{n}</p>" for n in notes)}</div>
<p class="foot">Dane tablicy: benchmarkheaven.com/jev-models, metoda v1.8.4, stan 31.10.2026. Fajfuzzy: Gzowo Labs,
github.com/JerzySukiennik/fajfuzzy.</p>
</body></html>"""


def render(page_path, png, height):
    tall = png.with_suffix(".tall.png")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
                    f"--force-device-scale-factor={SCALE}", f"--window-size={WIDTH},{height}",
                    f"--screenshot={tall}", page_path.resolve().as_uri()], check=True, capture_output=True)
    img = Image.open(tall).convert("RGB")
    bg = img.getpixel((2, img.height - 2))
    px = img.load()
    bottom = img.height
    while bottom > 1 and all(px[x, bottom - 1] == bg for x in range(0, img.width, 4)):
        bottom -= 1
    img.crop((0, 0, img.width, min(img.height, bottom + 48 * SCALE))).save(png, optimize=True)
    tall.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--board", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    bench, board = json.loads(a.bench.read_text()), json.loads(a.board.read_text())
    page_path = a.out.with_suffix(".html")
    page_path.write_text(page(bench, board), encoding="utf-8")
    height = 1300 + 13 * (len(board) + len(bench["same_items"])) + 400
    render(page_path, a.out, height)
    print(a.out, Image.open(a.out).size)


if __name__ == "__main__":
    main()
