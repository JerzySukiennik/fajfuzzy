"""Local Fajfuzzy demo server: type English text, the discriminator marks the words it thinks were swapped.

Serves the demo report page with a "try it yourself" box and answers POST /score. A short
input is scored between two real validation documents, so it sits in a full window the way
the model saw text in pretraining. Two CPU threads, idle between requests, keep the Mac quiet.

    python tools/rtd_server.py --ckpt disc.pt --prep prep --page fajfuzzy-podmiany-s2.html --port 8858
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtd_demo import Model, documents  # noqa: E402

CONTEXT = (2081, 2157)
EXAMPLES = ["I wood like a glass of water, please.", "We need to by more filament for the printer.",
            "The rocket was lunched from the field near the river.", "The printer finished the part in tree hours."]
FOOTER = ("Na pojedynczych, krótkich zdaniach jeszcze nie działa: w pretrainingu widzi tylko długie okna tekstu.",
          "Model zna z pretrainingu tylko długie okna tekstu, więc tekst z pola „Sprawdź sam” serwer wkłada między "
          "dwa prawdziwe akapity z walidacji. Bez tego na pojedynczym zdaniu nic by nie zaznaczył.")

BOX = """<section class="try">
<h2>Sprawdź sam</h2>
<p class="intro">Wpisz albo wklej tekst po angielsku. Fajfuzzy zaznaczy słowa, które jego zdaniem ktoś podmienił. Najlepiej łapie pomyłki typu „wood” zamiast „would”, a „lunched” zamiast „launched” jeszcze mu umyka.</p>
<form id="try-form">
<textarea id="try-text" rows="3" spellcheck="false" autocomplete="off">__FIRST__</textarea>
<div class="try-row">
<button id="try-go" type="submit">Sprawdź</button>
<span class="try-ex">__EXAMPLES__</span>
</div>
</form>
<div id="try-out" class="try-out" hidden><p class="s" id="try-res"></p><p class="meta" id="try-top"></p></div>
</section>
<style>
.try { margin:0 0 56px; }
.try textarea { display:block; width:100%; font:18px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif; color:var(--ink); background:#fff; border:1px solid #d2d2d7; border-radius:14px; padding:14px 16px; resize:vertical; outline:none; }
.try textarea:focus { border-color:var(--ink); }
.try-row { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:12px 0 0; }
#try-go { font:600 16px/1 -apple-system, BlinkMacSystemFont, sans-serif; color:#fff; background:var(--ink); border:0; border-radius:999px; padding:12px 22px; cursor:pointer; }
#try-go:disabled { opacity:.45; cursor:default; }
.try-ex { display:flex; flex-wrap:wrap; gap:6px; }
.ex { font:13px/1.25 -apple-system, BlinkMacSystemFont, sans-serif; color:var(--ink); background:var(--card); border:0; border-radius:999px; padding:7px 12px; cursor:pointer; text-align:left; }
.try-out { background:var(--card); border-radius:16px; padding:18px 20px; margin:16px 0 0; }
.try-out .s { white-space:pre-wrap; line-height:1.7; }
</style>
<script>
(() => {
  const form = document.getElementById('try-form'), box = document.getElementById('try-text'), go = document.getElementById('try-go');
  const out = document.getElementById('try-out'), res = document.getElementById('try-res'), top = document.getElementById('try-top');
  const esc = s => s.replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const pct = p => Math.round(p * 100) + '%';
  async function run() {
    const text = box.value.trim();
    if (!text || go.disabled) return;
    go.disabled = true;
    go.textContent = 'Liczę…';
    try {
      const r = await fetch('/score', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
      const d = await r.json();
      let h = '', pos = 0;
      for (const w of d.words) {
        h += esc(d.text.slice(pos, w.start));
        const bg = w.p >= 0.12 ? ` style="background:rgba(255,184,0,${(Math.min(1, w.p) * 0.85).toFixed(2)})"` : '';
        h += `<span class="w"${bg} title="podejrzenie ${pct(w.p)}">${esc(d.text.slice(w.start, w.end))}</span>`;
        pos = w.end;
      }
      res.innerHTML = h + esc(d.text.slice(pos));
      const best = d.words.filter(w => /[A-Za-z0-9]/.test(d.text.slice(w.start, w.end))).sort((a, b) => b.p - a.p).slice(0, 3);
      top.textContent = 'najbardziej podejrzane: ' + best.map(w => `„${d.text.slice(w.start, w.end)}” ${pct(w.p)}`).join(' · ') + ` · ${d.ms} ms`;
    } catch (e) {
      res.textContent = '';
      top.textContent = 'Serwer demo nie odpowiada. Pole działa tylko na stronie otwartej przez serwer Fajfuzzy.';
    }
    out.hidden = false;
    go.disabled = false;
    go.textContent = 'Sprawdź';
  }
  form.addEventListener('submit', e => { e.preventDefault(); run(); });
  box.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(); } });
  document.querySelectorAll('.ex').forEach(b => b.addEventListener('click', () => { box.value = b.textContent; run(); }));
})();
</script>
"""


class Scorer:
    def __init__(self, model, val):
        self.m, self.lock = model, threading.Lock()
        docs = documents(val, model.cfg.sep_id)
        self.before = [int(t) for t in docs[CONTEXT[0]][-250:]]
        self.after = [int(t) for t in docs[CONTEXT[1]][:250]]

    def __call__(self, text):
        m = self.m
        ids = m.tok.encode(text.rstrip() + "\n").ids[:500]
        if not ids:
            return {"text": "", "words": [], "ms": 0}
        room = max(0, 509 - len(ids))
        k = min(room // 2, len(self.before))
        a, b = self.before[len(self.before) - k:], self.after[:room - k]
        seq = [m.cfg.cls_id] + a + [m.cfg.sep_id] + ids + [m.cfg.sep_id] + b
        t = time.time()
        with self.lock:
            p = m.rtd(torch.tensor([seq]))[0].tolist()
        off = 2 + len(a)
        decoded, ws = m.words(ids, p[off:off + len(ids)])
        return {"text": decoded, "words": [{"start": w["start"], "end": w["end"], "p": round(w["p"], 4)} for w in ws],
                "ms": round((time.time() - t) * 1000)}


def make_handler(page, score):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, kind):
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
            if self.path != "/score":
                return self.send(404, b"not found", "text/plain")
            n = min(int(self.headers.get("Content-Length", 0)), 20000)
            text = str(json.loads(self.rfile.read(n) or b"{}").get("text", ""))[:3000]
            self.send(200, json.dumps(score(text)).encode(), "application/json")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="ckpt.pt or a disc.pt export")
    ap.add_argument("--prep", type=Path, required=True, help="folder with tokenizer.json and decide_val.bin")
    ap.add_argument("--page", type=Path, required=True, help="report page from tools/rtd_demo.py")
    ap.add_argument("--port", type=int, default=8858)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    m = Model(a.ckpt, a.prep / "tokenizer.json", a.threads)
    score = Scorer(m, np.memmap(a.prep / "decide_val.bin", dtype=np.uint16, mode="r"))
    chips = "".join(f'<button type="button" class="ex">{e}</button>' for e in EXAMPLES[:-1])
    box = BOX.replace("__FIRST__", EXAMPLES[-1]).replace("__EXAMPLES__", chips)
    page = a.page.read_text(encoding="utf-8").replace("<section>", box + "\n<section>", 1).replace(*FOOTER).encode()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(page, score))
    print(f"Fajfuzzy demo (step {m.step}) on http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
