#!/usr/bin/env python3
"""Run Fajfuzzy's pretraining unattended: Kaggle sessions one after another, each resuming
the previous checkpoint, until the run reaches MAX_STEPS. Adapted from G-Dev's chain.

The loop is stateless: every turn asks Kaggle what exists and derives the next move, so a
reboot loses nothing. tools/state.json only keeps the measured quota factor and cached
step counts. A session must end on its own (--max-hours) before the quota runs out,
because Kaggle refuses a cancelled kernel as the source of the next one. Every wait is
capped at 30 minutes and runs as one-minute naps against the wall clock, because
time.sleep does not count the Mac's sleep: after a wake the chain acts within a minute.
Nothing starts until the prep kernel has finished, so the chain can be installed early.

    LaunchAgent fun.gzowo.fajfuzzy.chain  (tools/chain-install.sh)
    stop: touch tools/STOP
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path.home() / "Downloads/Claude/Projects/AIe/Fajfuzzy"
KAGGLE = [str(Path.home() / "Downloads/Claude/Projects/AIe/G-Images/.venv/bin/python"), "-m", "kaggle"]
CFG = Path.home() / ".kaggle-gdev"
USER = (CFG / "username").read_text().strip()
KAGGLE_ENV = {**os.environ, "KAGGLE_API_TOKEN": str(CFG / "access_token")}

PREFIX, PREP = "fajfuzzy-s", "gdec-prep"
MAX_SESSION = 40
MAX_STEPS = 36000
STATE = REPO / "tools/state.json"
STOP_FILE = REPO / "tools/STOP"
DONE_FILE = REPO / "tools/DONE"
TMP = REPO / "Niepotrzebne/chain-tmp"

BUSY = ("RUNNING", "QUEUED", "NEW_SCRIPT", "CANCEL_REQUESTED")
FAILED = ("ERROR", "CANCEL_ACKNOWLEDGED")

POLL = 900
PREP_POLL = 300
WAIT_CAP = 1800
RETRY_WAIT = 1800
MAX_FAILS = 4
SESSION_CAP = 11.0
SETUP_MARGIN = 0.6
MIN_USEFUL = 0.8


def log(msg):
    print(f"[{datetime.now():%F %H:%M}] {msg}", flush=True)


def nap(seconds):
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(min(60, max(1, end - time.time())))


def notify(msg):
    log(f"POWIADOMIENIE: {msg}")
    subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "Fajfuzzy"'], capture_output=True)


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"factor": 1.05}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1))


def kaggle(*args, timeout=300):
    try:
        r = subprocess.run(KAGGLE + list(args), capture_output=True, text=True, timeout=timeout, env=KAGGLE_ENV)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def status(name):
    m = re.search(r"KernelWorkerStatus\.(\w+)", kaggle("kernels", "status", f"{USER}/{name}"))
    return m.group(1) if m else "NONE"


def quota():
    m = re.search(r"^GPU\s+([\d.]+)h\s+([\d.]+)h\s+[\d.]+h\s+(\S+)", kaggle("quota"), re.M)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), datetime.fromisoformat(m.group(3)).replace(tzinfo=timezone.utc)


def until_refresh(q):
    return max(900, (q[2] - datetime.now(timezone.utc)).total_seconds() + 600)


def session_hours(factor):
    q = quota()
    if q is None:
        log("cannot read the quota, taking a careful 3 h")
        return 3.0
    h = min((q[1] - SETUP_MARGIN) / factor, SESSION_CAP)
    return 0.0 if h < MIN_USEFUL else round(h, 1)


def push_kernel(name, code_file, sources, replace):
    d = REPO / f"Niepotrzebne/kernel-{name}"
    d.mkdir(parents=True, exist_ok=True)
    code = (REPO / code_file).read_text(encoding="utf-8")
    for pat, val in replace.items():
        code = re.sub(pat, val, code, count=1, flags=re.M)
    (d / "kernel.py").write_text(code, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{name}", "title": name, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": sources, "model_sources": [],
    }, indent=2), encoding="utf-8")
    out = kaggle("kernels", "push", "-p", str(d), timeout=900)
    log(f"push {name}: {out.strip().splitlines()[-1] if out.strip() else '(no output)'}")
    if "Maximum batch GPU session count" in out:
        return 3
    return 0 if "successfully pushed" in out else 1


def launch(prev, nxt, st):
    hours = session_hours(st["factor"])
    if hours == 0.0:
        log(f"quota nearly gone, not starting {nxt}")
        return 2
    q = quota()
    rc = push_kernel(nxt, "kaggle/02-pretrain.py", [f"{USER}/{PREP}"] + ([f"{USER}/{prev}"] if prev else []),
                     {r"^SESSION_HOURS = .*$": f"SESSION_HOURS = {hours}",
                      r"^EXPECT_RESUME = .*$": f"EXPECT_RESUME = {bool(prev)}"})
    if rc == 0:
        log(f"{nxt} gets {hours} h (quota factor {st['factor']:.2f})")
        st["launch"] = {"name": nxt, "hours": hours, "used": q[0] if q else None, "ts": time.time(), "measured": False}
        save_state(st)
    return rc


def calibrate(st, sess):
    ln = st.get("launch")
    if not ln or ln.get("measured") or ln.get("used") is None:
        return
    if sess.get(int(ln["name"].replace(PREFIX, ""))) != "COMPLETE":
        return
    q = quota()
    if q is None or q[0] < ln["used"]:
        return
    measured = (q[0] - ln["used"]) / (ln["hours"] + 0.25)
    st["factor"] = round(min(2.0, max(1.0, measured * 1.05)), 2)
    ln["measured"] = True
    log(f"{ln['name']}: used {q[0] - ln['used']:.2f} h of quota for a {ln['hours']} h session -> factor {st['factor']}")
    save_state(st)


def last_step(name, st):
    steps = st.setdefault("steps", {})
    if name in steps:
        return steps[name]
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{USER}/{name}", "-p", str(d), "--file-pattern", r"log\.jsonl$", timeout=600)
    path = next(d.rglob("log.jsonl"), None)
    if not path:
        return None
    step = 0
    for line in path.read_text().splitlines():
        try:
            step = max(step, json.loads(line).get("step", 0))
        except ValueError:
            pass
    steps[name] = step
    save_state(st)
    log(f"{name} ended at step {step}/{MAX_STEPS}")
    return step


def finish(last_ok):
    out = REPO / "Niepotrzebne/pretrain-final"
    out.mkdir(parents=True, exist_ok=True)
    log("pretraining finished, downloading the discriminator")
    kaggle("kernels", "output", f"{USER}/{last_ok}", "-p", str(out), "--file-pattern",
           r"(disc\.pt|tokenizer\.json|log\.jsonl)$", timeout=3600)
    disc = next(out.rglob("disc.pt"), None)
    DONE_FILE.write_text(f"session: {last_ok}\ndiscriminator: {disc}\n")
    notify("Fajfuzzy: pretraining skonczony" + ("" if disc else ", ale disc.pt sie nie pobral"))


def main():
    st = load_state()
    log("chain started")
    ignore = st.get("ignore", 0)
    while True:
        if STOP_FILE.exists():
            log("STOP found, exiting")
            return 0
        prep = status(PREP)
        if prep in FAILED:
            notify(f"{PREP} skonczyl sie statusem {prep}, lancuch zatrzymany")
            STOP_FILE.touch()
            return 0
        if prep != "COMPLETE":
            log(f"waiting for {PREP} ({prep})")
            nap(PREP_POLL)
            continue
        top =min(max(4, st.get("top", 0) + 3), MAX_SESSION)
        sess = {n: status(f"{PREFIX}{n}") for n in range(1, top + 1)}
        existing = [n for n, s in sess.items() if s != "NONE"]
        st["top"] = max(existing, default=0)
        calibrate(st, sess)
        if any(s in BUSY for s in sess.values()):
            nap(POLL)
            continue

        done = [n for n, s in sess.items() if s == "COMPLETE"]
        last_ok = f"{PREFIX}{max(done)}" if done else None
        if last_ok and (last_step(last_ok, st) or 0) >= MAX_STEPS:
            finish(last_ok)
            return 0
        if len(done) >= 2:
            prev_ok = f"{PREFIX}{sorted(done)[-2]}"
            if (last_step(last_ok, st) or 0) <= (last_step(prev_ok, st) or 0):
                notify(f"{last_ok} nie posunal treningu dalej niz {prev_ok}, lancuch zatrzymany")
                STOP_FILE.touch()
                return 0

        base = max(done, default=0)
        fails = sum(1 for n, s in sess.items() if n > max(base, ignore) and s in FAILED)
        if fails >= MAX_FAILS:
            q = quota()
            wait = until_refresh(q) if q else 6 * 3600
            notify(f"{fails} nieudane starty z rzedu, ponawiam po odnowieniu quoty (za {wait / 3600:.1f} h)")
            ignore = st["ignore"] = st["top"]
            save_state(st)
            nap(min(wait, WAIT_CAP))
            continue
        if fails:
            left = RETRY_WAIT - (time.time() - st.get("launch", {}).get("ts", 0))
            if left > 0:
                nap(min(left, WAIT_CAP))
                continue

        rc = launch(last_ok, f"{PREFIX}{max(existing, default=0) + 1}", st)
        if rc == 0:
            nap(180)
        elif rc == 2:
            q = quota()
            nap(min(until_refresh(q) if q else 3600, WAIT_CAP))
        elif rc == 3:
            nap(600)
        else:
            nap(RETRY_WAIT)


if __name__ == "__main__":
    sys.exit(main())
