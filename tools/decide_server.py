"""Local demo of Fajfuzzy's decision model: a question with answers gets a probability per answer,
and a step instruction picks a control of a macOS window the way Concierge's agent mode will.

Two tabs on one page, POST /decide and POST /click. Two CPU threads, idle between requests.

    python tools/decide_server.py --model decider.pt --tokenizer tokenizer.json --port 8859
"""

import argparse
import html
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.decider import Decider, encode, option_probs  # noqa: E402

CLICK_AT = 0.85

SAVE = ['text field "Save As"', 'pop up button "Where"', 'checkbox "Hide extension"', 'button "New Folder"',
        'button "Cancel"', 'button "Save"']
WIFI = ['checkbox "Wi-Fi"', 'button "Details…"', 'button "Other…"', 'pop up button "Ask to join networks"',
        'button "Advanced…"', 'button "?"']
DEPTS = ["Billing", "Shipping", "Technical support", "Sales"]

DECIDE_GUIDE = [
    ("Jasny sentyment", "Recenzja z mocnymi słowami i dwie odpowiedzi.",
     "positive ok. 92%. To jedna z rzeczy, które v1 robi pewnie. Zła recenzja daje negative ok. 66%.",
     {"question": "What is the sentiment of this review?", "options": ["positive", "negative"],
      "state": "I love it, best purchase ever!"}),
    ("Czy jedno wynika z drugiego", "Dwa zdania, pytanie sformułowane tak jak w treningu.",
     "contradiction ok. 66%, entailment prawie zero. Inne sformułowanie pytania potrafi zbić pewność do ok. 45%.",
     {"question": "Does text_A entail text_B, contradict it, or neither?", "options": ["entailment", "neutral", "contradiction"],
      "state": "text_A: The rocket launched from the field at noon and reached 300 meters.\ntext_B: The rocket never left the ground."}),
    ("Do którego działu", "Wiadomość od klienta i lista działów.",
     "Sales ok. 68%. Zmień tekst na „The app crashes every time I open the settings page.” (Technical support ok. 51%), "
     "a potem na spóźnioną paczkę: tu się myli i wskazuje Sales zamiast Shipping.",
     {"question": "Which department should handle this message?", "options": DEPTS,
      "state": "Do you offer a discount if we buy 50 licenses for our school?"}),
    ("Is a hotdog a sub?", "Pytanie z opinii, bez kontekstu.",
     "Yes ok. 53%. „Is cereal a soup?” i „Is water wet?” też wychodzą blisko 50/50: bez kontekstu v1 daje tak prawie każdemu "
     "pytaniu, więc to bardziej niewiedza niż prawdziwy rozkład opinii.",
     {"question": "Is a hotdog a sub?", "options": ["Yes", "No"], "state": ""}),
    ("Pułapka: oczywiste tak/nie", "Ewidentny spam i pytanie tak/nie.",
     "spam dostaje tylko ok. 62% Yes, a zwykła wiadomość od kolegi też ok. 53%. Tak/nie to najsłabsza część v1: w danych "
     "pytania tak/nie to głównie trudne łamigłówki, a proste klasyfikacje mają nazwane odpowiedzi.",
     {"question": "Is this message spam?", "options": ["Yes", "No"],
      "state": "Congratulations! You won a free iPhone, click here to claim your prize now."}),
]
CLICK_GUIDE = [
    ("Dosłowne polecenie", "Nazwa elementu wprost w poleceniu.",
     "button „Cancel” ponad 99%, kliknąłby sam. Tak samo „Make a new folder” trafia w „New Folder”.",
     {"instruction": "Click Cancel", "app": "TextEdit", "window": "Untitled", "elements": SAVE}),
    ("Parafraza", "Polecenie innymi słowami niż nazwa elementu.",
     "przełącznik „Wi-Fi” ponad 99%. W Safari „Open a new tab” trafia w „New Tab”, a „Go back to the previous page” w „Back” (91%).",
     {"instruction": "Turn off Wi-Fi", "app": "System Settings", "window": "Wi-Fi", "elements": WIFI}),
    ("Kiedy nie wie", "Polecenie wpisania tekstu: w treningu były tylko kliknięcia.",
     "rozrzut, najwyżej ok. 55% na złym elemencie, więc krok wraca do Gemini. Tak ma działać próg pewności (85%).",
     {"instruction": "Type the file name", "app": "TextEdit", "window": "Untitled", "elements": SAVE}),
    ("Pułapka: przeczenie", "„Nie zapisuj, zamknij okno”, po angielsku.",
     "klika „Save” z pewnością ponad 99%. Model dopasowuje słowa i nie rozumie „don't”. Przed podpięciem do Concierge trzeba go tego nauczyć.",
     {"instruction": "Don't save, close the dialog", "app": "TextEdit", "window": "Untitled", "elements": SAVE}),
    ("Pułapka: brak elementu", "Polecenie elementu, którego w oknie nie ma.",
     "wybiera „Wi-Fi” z pewnością ok. 90%, czyli kliknąłby. Nie umie powiedzieć „nie ma takiego”, bo w treningu właściwy element był zawsze.",
     {"instruction": "Click the Bluetooth button", "app": "System Settings", "window": "Wi-Fi", "elements": WIFI}),
]

PAGE = """<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fajfuzzy · decyzje</title>
<style>
:root { --ink:#1d1d1f; --mute:#6e6e73; --line:#e5e5ea; --card:#f5f5f7; --bar:#d2d2d7; }
* { box-sizing:border-box; }
body { margin:0; background:#fff; color:var(--ink); font:17px/1.55 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; -webkit-font-smoothing:antialiased; }
main { max-width:780px; margin:0 auto; padding:56px 16px 72px; }
.kicker { font:600 12px/1.2 ui-monospace, "SF Mono", Menlo, monospace; letter-spacing:.06em; text-transform:uppercase; color:var(--mute); margin:0 0 14px; }
h1 { font:700 42px/1.06 -apple-system, "SF Pro Display", sans-serif; letter-spacing:-.025em; margin:0 0 16px; }
.lede { font-size:19px; color:var(--mute); margin:0 0 36px; max-width:640px; }
.stats { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin:0 0 12px; }
.stat { background:var(--card); border-radius:16px; padding:16px 16px 14px; }
.stat b { display:block; font:700 32px/1 -apple-system, "SF Pro Display", sans-serif; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
.stat span { display:block; margin-top:8px; font-size:13px; line-height:1.35; color:var(--mute); }
.note { font-size:14px; color:var(--mute); margin:0 0 48px; }
section { margin:0 0 56px; }
h2 { font:700 27px/1.2 -apple-system, "SF Pro Display", sans-serif; letter-spacing:-.015em; margin:0 0 12px; }
.intro { color:var(--mute); margin:0 0 18px; }
.seg { display:inline-flex; background:var(--card); border-radius:999px; padding:4px; margin:0 0 22px; }
.seg button { font:600 15px/1 -apple-system, BlinkMacSystemFont, sans-serif; color:var(--mute); background:none; border:0; border-radius:999px; padding:10px 18px; cursor:pointer; }
.seg button[aria-selected="true"] { color:var(--ink); background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.08); }
label { display:block; font:600 13px/1.3 -apple-system, BlinkMacSystemFont, sans-serif; color:var(--mute); margin:14px 0 6px; }
label small { font-weight:400; }
input, textarea { display:block; width:100%; font:17px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif; color:var(--ink); background:#fff; border:1px solid var(--bar); border-radius:14px; padding:12px 14px; outline:none; }
textarea { resize:vertical; }
textarea.mono { font:15px/1.5 ui-monospace, "SF Mono", Menlo, monospace; }
input:focus, textarea:focus { border-color:var(--ink); }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.go { font:600 16px/1 -apple-system, BlinkMacSystemFont, sans-serif; color:#fff; background:var(--ink); border:0; border-radius:999px; padding:13px 24px; margin:16px 0 0; cursor:pointer; }
.go:disabled { opacity:.45; cursor:default; }
.out { background:var(--card); border-radius:16px; padding:18px 20px; margin:16px 0 0; }
.verdict { font:600 17px/1.4 -apple-system, "SF Pro Display", sans-serif; margin:0 0 12px; }
.rows { display:grid; gap:12px; }
.row .top { display:flex; justify-content:space-between; gap:12px; font-size:15px; }
.row .top span:last-child { font-variant-numeric:tabular-nums; color:var(--mute); }
.row.best .top { font-weight:600; }
.row.best .top span:last-child { color:var(--ink); }
.track { height:8px; background:#fff; border-radius:999px; margin:6px 0 0; overflow:hidden; }
.fill { height:100%; width:0; background:var(--bar); border-radius:999px; transition:width 240ms cubic-bezier(.165, .84, .44, 1); }
.row.best .fill { background:var(--ink); }
.meta { font-size:13px; color:var(--mute); margin:12px 0 0; }
.steps { list-style:none; padding:0; margin:0 0 18px; display:grid; gap:10px; }
.steps li { display:grid; grid-template-columns:32px 1fr; gap:4px 12px; background:var(--card); border-radius:16px; padding:16px 18px; }
.steps .n { font:700 20px/1.3 -apple-system, "SF Pro Display", sans-serif; color:var(--mute); font-variant-numeric:tabular-nums; }
.steps h3 { font:600 17px/1.3 -apple-system, "SF Pro Display", sans-serif; margin:0 0 2px; }
.steps p { margin:0; font-size:15px; }
.steps .good { color:var(--mute); margin:4px 0 0; }
.ex { font:13px/1.25 -apple-system, BlinkMacSystemFont, sans-serif; color:var(--ink); background:#fff; border:0; border-radius:999px; padding:7px 12px; margin:10px 0 0; cursor:pointer; transition:transform 120ms cubic-bezier(.165, .84, .44, 1); }
.ex:active { transform:scale(.97); }
@media (max-width:600px) { h1 { font-size:34px; } .stats { grid-template-columns:repeat(2, 1fr); } .pair { grid-template-columns:1fr; } }
@media (prefers-reduced-motion:reduce) { .fill, .ex { transition:none; } }
</style>
</head>
<body>
<main>
<p class="kicker">Fajfuzzy · doszkolenie v1 · krok __STEP__</p>
<h1>Decyzje i klikanie</h1>
<p class="lede">Ten sam enkoder 110M co w demie podmienionych słów, doszkolony na prawie milionie decyzji z ponad 500 zadań i 42 tysiącach kroków klikania w prawdziwych oknach macOS. Każda odpowiedź dostaje procent, procenty sumują się do 100.</p>
<div class="stats">__STATS__</div>
<p class="note">Tam, gdzie ludzie się kłócą, uczył się na ich głosach, więc procent ma znaczyć rozkład opinii. W v1 wynik blisko 50/50 to jednak częściej niewiedza niż spór. Działa tylko po angielsku.</p>
<section>
<div class="seg" role="tablist">
<button type="button" role="tab" data-tab="decide" aria-selected="true">Decyzje</button>
<button type="button" role="tab" data-tab="click" aria-selected="false">Elementy</button>
</div>
<form id="panel-decide" class="panel">
<label for="d-q">Pytanie</label>
<input id="d-q" autocomplete="off" spellcheck="false" value="Is a hotdog a sub?">
<label for="d-o">Odpowiedzi <small>(każda w osobnej linii, 2-12)</small></label>
<textarea id="d-o" rows="3" spellcheck="false">Yes
No</textarea>
<label for="d-s">Kontekst <small>(opcjonalnie: tekst, którego dotyczy pytanie)</small></label>
<textarea id="d-s" rows="3" spellcheck="false"></textarea>
<button class="go" type="submit">Policz</button>
<div class="out" hidden><div class="rows"></div><p class="meta"></p></div>
</form>
<form id="panel-click" class="panel" hidden>
<label for="c-q">Polecenie <small>(jeden krok, tak jak Gemini poda go Fajfuzzy)</small></label>
<input id="c-q" autocomplete="off" spellcheck="false" value="Click the Save button">
<div class="pair">
<div><label for="c-a">Apka</label><input id="c-a" autocomplete="off" spellcheck="false" value="TextEdit"></div>
<div><label for="c-w">Okno</label><input id="c-w" autocomplete="off" spellcheck="false" value="Untitled"></div>
</div>
<label for="c-e">Elementy okna <small>(rola i nazwa, każdy w osobnej linii)</small></label>
<textarea id="c-e" class="mono" rows="7" spellcheck="false">text field "Save As"
pop up button "Where"
checkbox "Hide extension"
button "New Folder"
button "Cancel"
button "Save"</textarea>
<button class="go" type="submit">Wybierz element</button>
<div class="out" hidden><p class="verdict"></p><div class="rows"></div><p class="meta"></p></div>
</form>
</section>
<section>
<h2>Jak testować</h2>
<p class="intro">„Wstaw i sprawdź” przełącza zakładkę, wkleja przykład i od razu go liczy. Potem zmieniaj go po swojemu.</p>
<h3 class="kicker">Decyzje</h3>
<ol class="steps">__DECIDE__</ol>
<h3 class="kicker">Elementy</h3>
<ol class="steps">__CLICK__</ol>
</section>
</main>
<script>
(() => {
  const $ = id => document.getElementById(id);
  const tabs = document.querySelectorAll('.seg button'), panels = document.querySelectorAll('.panel');
  const lines = s => s.split('\\n').map(x => x.trim()).filter(Boolean);
  const pct = p => (p >= 0.995 ? '>99' : p < 0.005 ? '<1' : Math.round(p * 100)) + '%';
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  function show(tab) {
    tabs.forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
    panels.forEach(p => p.hidden = p.id !== 'panel-' + tab);
  }
  function draw(form, names, probs, sort) {
    const best = probs.indexOf(Math.max(...probs));
    let order = names.map((_, i) => i);
    if (sort) order.sort((a, b) => probs[b] - probs[a]);
    const rows = form.querySelector('.rows');
    rows.replaceChildren(...order.map(i => {
      const r = document.createElement('div');
      r.className = 'row' + (i === best ? ' best' : '');
      r.innerHTML = '<div class="top"><span></span><span></span></div><div class="track"><div class="fill"></div></div>';
      r.querySelector('.top span').textContent = names[i];
      r.querySelector('.top span:last-child').textContent = pct(probs[i]);
      r.dataset.w = (probs[i] * 100).toFixed(1) + '%';
      return r;
    }));
    requestAnimationFrame(() => requestAnimationFrame(() => rows.querySelectorAll('.row').forEach(r => r.querySelector('.fill').style.width = r.dataset.w)));
    return best;
  }
  async function run(form) {
    const go = form.querySelector('.go'), out = form.querySelector('.out'), meta = form.querySelector('.meta');
    const click = form.id === 'panel-click';
    const body = click
      ? {instruction: $('c-q').value.trim(), app: $('c-a').value.trim(), window: $('c-w').value.trim(), elements: lines($('c-e').value)}
      : {question: $('d-q').value.trim(), state: $('d-s').value.trim(), options: lines($('d-o').value)};
    const names = click ? body.elements : body.options;
    if (go.disabled) return;
    out.hidden = false;
    if (names.length < 2 || names.length > 12 || !(click ? body.instruction : body.question)) {
      form.querySelector('.rows').replaceChildren();
      if (click) form.querySelector('.verdict').textContent = '';
      meta.textContent = click ? 'Podaj polecenie i od 2 do 12 elementów.' : 'Podaj pytanie i od 2 do 12 odpowiedzi.';
      return;
    }
    go.disabled = true;
    const label = go.textContent;
    go.textContent = 'Liczę…';
    try {
      const r = await fetch(click ? '/click' : '/decide', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'błąd');
      const best = draw(form, names, d.probs, click);
      if (click) form.querySelector('.verdict').textContent = d.click
        ? `Kliknąłby sam: ${names[best]}`
        : `Za mało pewny (najwyżej ${pct(d.probs[best])}), w Concierge ten krok wróciłby do Gemini`;
      meta.textContent = `${d.tokens} tokenów · ${d.ms} ms`;
    } catch (e) {
      form.querySelector('.rows').replaceChildren();
      if (click) form.querySelector('.verdict').textContent = '';
      meta.textContent = e.message === 'Failed to fetch' ? 'Serwer demo nie odpowiada. Strona działa tylko otwarta przez serwer Fajfuzzy.' : e.message;
    }
    go.disabled = false;
    go.textContent = label;
  }
  tabs.forEach(b => b.addEventListener('click', () => show(b.dataset.tab)));
  panels.forEach(f => {
    f.addEventListener('submit', e => { e.preventDefault(); run(f); });
    f.querySelectorAll('input').forEach(i => i.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); run(f); } }));
  });
  document.querySelectorAll('.ex').forEach(b => b.addEventListener('click', () => {
    const x = JSON.parse(b.dataset.x), tab = b.dataset.tab;
    show(tab);
    if (tab === 'click') { $('c-q').value = x.instruction; $('c-a').value = x.app; $('c-w').value = x.window; $('c-e').value = x.elements.join('\\n'); }
    else { $('d-q').value = x.question; $('d-o').value = x.options.join('\\n'); $('d-s').value = x.state || ''; }
    document.querySelector('.seg').scrollIntoView({behavior: still.matches ? 'auto' : 'smooth', block: 'start'});
    run($('panel-' + tab));
  }));
})();
</script>
</body>
</html>
"""


class Model:
    def __init__(self, path, tokenizer, threads):
        torch.set_num_threads(threads)
        ck = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
        self.step, self.meta = ck["step"], ck.get("meta") or {}
        self.net = Decider.from_file(path).eval()
        self.tok = Tokenizer.from_file(str(tokenizer))
        self.lock = threading.Lock()

    @torch.no_grad()
    def decide(self, question, state, options):
        ids = encode(self.tok, question, state, options, None, 512)
        t = time.time()
        with self.lock:
            x = torch.tensor([ids])
            p = option_probs(self.net(x, torch.tensor([len(ids)])), x)[0].tolist()
        return {"probs": [round(v, 4) for v in p], "tokens": len(ids), "ms": round((time.time() - t) * 1000)}

    def click(self, instruction, app, window, elements):
        state = f"App: {app or 'unknown'}\nWindow: {window or 'untitled'}\n" + "\n".join(elements)
        out = self.decide(instruction, state, elements)
        out["click"] = max(out["probs"]) >= CLICK_AT
        return out


def gui_accuracy(model_path, step):
    log = Path(model_path).with_name("log.jsonl")
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        rec = json.loads(line)
        if rec["step"] == step:
            gui = [x for k, x in rec["val"].items() if k.startswith("gui/")]
            return sum(x["acc"] * x["n"] for x in gui) / sum(x["n"] for x in gui)
    return None


def build_page(m, model_path):
    gui = gui_accuracy(model_path, m.step)
    stats = [(f"{gui:.0%}" if gui is not None else "?", "trafionych elementów w apkach, których nie widział w treningu"),
             (f"{m.meta.get('val_mean_acc', 0):.0%}", "trafień średnio na 541 zadaniach decyzyjnych"),
             (f"{m.meta.get('jev_acc') or 0:.0%}", "na JevBench (cel 50-60%, Jev ma 87%)"),
             ("110M", "parametrów, liczy na 2 wątkach CPU")]
    cards = "".join(f'<div class="stat"><b>{html.escape(b)}</b><span>{html.escape(s)}</span></div>' for b, s in stats)

    def steps(guide, tab):
        return "".join(
            f'<li><div class="n">{i}</div><div><h3>{html.escape(title)}</h3><p>{html.escape(what)}</p>'
            f'<p class="good">Czego się spodziewać: {html.escape(good)}</p>'
            f'<button type="button" class="ex" data-tab="{tab}" data-x="{html.escape(json.dumps(x))}">Wstaw i sprawdź</button></div></li>'
            for i, (title, what, good, x) in enumerate(guide, 1))

    return (PAGE.replace("__STEP__", f"{m.step:,}".replace(",", " ")).replace("__STATS__", cards)
            .replace("__DECIDE__", steps(DECIDE_GUIDE, "decide")).replace("__CLICK__", steps(CLICK_GUIDE, "click")).encode())


def texts(value, cap, limit):
    return [str(x).strip()[:limit] for x in (value or []) if str(x).strip()][:cap]


def make_handler(page, m):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                self.send(200, page, "text/html; charset=utf-8")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path not in ("/decide", "/click"):
                return self.send(404, b"not found", "text/plain")
            n = min(int(self.headers.get("Content-Length", 0)), 40000)
            d = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/decide":
                    options = texts(d.get("options"), 12, 200)
                    if len(options) < 2:
                        raise ValueError("Podaj co najmniej dwie odpowiedzi.")
                    out = m.decide(str(d.get("question", ""))[:600], str(d.get("state", ""))[:8000], options)
                else:
                    elements = texts(d.get("elements"), 12, 200)
                    if len(elements) < 2:
                        raise ValueError("Podaj co najmniej dwa elementy.")
                    out = m.click(str(d.get("instruction", ""))[:300], str(d.get("app", ""))[:80],
                                  str(d.get("window", ""))[:120], elements)
            except ValueError as e:
                msg = str(e) if str(e).startswith("Podaj") else "Odpowiedzi są za długie, żeby zmieścić się w 512 tokenach."
                return self.send(400, json.dumps({"error": msg}).encode())
            self.send(200, json.dumps(out).encode())

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="decider.pt from train/finetune.py")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8859)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    m = Model(a.model, a.tokenizer, a.threads)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(build_page(m, a.model), m))
    print(f"Fajfuzzy decisions (step {m.step}) on http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
