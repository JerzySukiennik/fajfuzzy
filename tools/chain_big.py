#!/usr/bin/env python3
"""Run the grown Fajfuzzy's pretraining across both Kaggle accounts until MAX_STEPS.

Same idea as tools/chain.py (one turn per GitHub Actions run, state on a branch, every turn
derives the next move from what Kaggle reports), with two accounts: a session runs where the
previous one ran while that account has quota, then moves to the other. A session's output
can only be mounted on its own account, so on a move the checkpoint is downloaded and
uploaded to the new account as the dataset fajfuzzy-big-ckpt first, and the session refuses
to resume from anything older (EXPECT_STEP). Nothing starts on an account while any other
kernel runs there (fine-tuning, data prep, other projects), and the session length comes
from the account's real remaining quota.

    python tools/chain_big.py --once --state state/state.json
    stop: a STOP file next to the state file, or disable the workflow
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = Path.home() / "Downloads/Claude/Projects/AIe/G-Images/.venv/bin/python"
KAGGLE = [str(VENV) if VENV.exists() else sys.executable, "-m", "kaggle"]
MAIN_CFG = Path.home() / (".kaggle-main" if (Path.home() / ".kaggle-main").exists() else ".kaggle")
ACCOUNTS = {
    "jerzysukiennik": {"token": MAIN_CFG / "access_token", "prep": "fajfuzzy-big-prep"},
    "jerzysukienik": {"token": Path.home() / ".kaggle-gdev/access_token", "prep": "gdec-prep"},
}
ORDER = list(ACCOUNTS)

PREFIX = "fajfuzzy-big-s"
INIT, CARRY = "fajfuzzy-big-init", "fajfuzzy-big-ckpt"
LAUNCHER = "kaggle/04-pretrain-big.py"
MAX_STEPS = 44000
MAX_SESSION = 40
STATE = REPO / "tools/state-big.json"
STOP_FILE = STATE.parent / "STOP"
DONE_FILE = STATE.parent / "DONE"
TMP = REPO / "Niepotrzebne/chain-big-tmp"
ONCE = False

BUSY = ("RUNNING", "QUEUED", "NEW_SCRIPT", "CANCEL_REQUESTED")
FAILED = ("ERROR", "CANCEL_ACKNOWLEDGED")

POLL = 900
WAIT_CAP = 1800
RETRY_WAIT = 1800
MAX_FAILS = 4
SESSION_CAP = 11.0
SETUP_MARGIN = 0.6
MIN_SESSION = 2.0
SWITCH_AFTER = 2700
CARRY_WAIT = 1500
CARRY_SETTLE = 300


class Pause(Exception):
    pass


def log(msg):
    line = f"[{datetime.now():%F %H:%M}] {msg}"
    print(line, flush=True)
    if ONCE:
        with (STATE.parent / "chain.log").open("a") as f:
            f.write(line + "\n")


def nap(seconds):
    if ONCE:
        raise Pause
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(min(60, max(1, end - time.time())))


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1))


def kaggle(acct, *args, timeout=300):
    env = {**os.environ, "KAGGLE_API_TOKEN": str(ACCOUNTS[acct]["token"])}
    try:
        r = subprocess.run(KAGGLE + list(args), capture_output=True, text=True, timeout=timeout, env=env)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def status(acct, name):
    m = re.search(r"KernelWorkerStatus\.(\w+)", kaggle(acct, "kernels", "status", f"{acct}/{name}"))
    return m.group(1) if m else "NONE"


def quota(acct):
    m = re.search(r"^GPU\s+([\d.]+)h\s+([\d.]+)h\s+[\d.]+h\s+(\S+)", kaggle(acct, "quota"), re.M)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), datetime.fromisoformat(m.group(3)).replace(tzinfo=timezone.utc)


def session_hours(acct, st):
    q = quota(acct)
    if q is None:
        return 0.0
    h = min((q[1] - SETUP_MARGIN) / st.setdefault("factor", {}).get(acct, 1.05), SESSION_CAP)
    return round(h, 1) if h >= MIN_SESSION else 0.0


def others_running(acct):
    """Kernels of this account other than the chain's that ran in the last 13 h and are still busy."""
    out = kaggle(acct, "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "20", "--csv")
    busy = []
    since = datetime.now(timezone.utc) - timedelta(hours=13)
    for row in csv.DictReader(io.StringIO(out[out.find("ref,"):] if "ref," in out else "")):
        ref = row.get("ref", "")
        try:
            ran = datetime.fromisoformat(row.get("lastRunTime", "")[:19]).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ran < since or ref.split("/")[-1].startswith(PREFIX):
            continue
        if status(acct, ref.split("/")[-1]) in BUSY:
            busy.append(ref)
    return busy


def sessions(st):
    where = st.setdefault("where", {})
    top = min(max(3, st.get("top", 0) + 3), MAX_SESSION)
    found = {}
    for n in range(1, top + 1):
        name = f"{PREFIX}{n}"
        for acct in ([where[name]] if name in where else ORDER):
            s = status(acct, name)
            if s != "NONE":
                found[n] = (acct, s)
                where[name] = acct
                break
    st["top"] = max(found, default=0)
    save_state(st)
    return found


def last_step(acct, name, st):
    steps = st.setdefault("steps", {})
    if name in steps:
        return steps[name]
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    kaggle(acct, "kernels", "output", f"{acct}/{name}", "-p", str(d), "--file-pattern", r"log\.jsonl$", timeout=600)
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
    log(f"{name} ({acct}) ended at step {step}/{MAX_STEPS}")
    return step


def calibrate(st, found):
    ln = st.get("launch")
    if not ln or ln.get("measured") or ln.get("used") is None:
        return
    n = int(ln["name"].replace(PREFIX, ""))
    if found.get(n, (None, None))[1] != "COMPLETE":
        return
    q = quota(ln["account"])
    if q is None or q[0] < ln["used"]:
        return
    measured = (q[0] - ln["used"]) / (ln["hours"] + 0.25)
    st.setdefault("factor", {})[ln["account"]] = round(min(2.0, max(1.0, measured * 1.05)), 2)
    ln["measured"] = True
    log(f"{ln['name']}: used {q[0] - ln['used']:.2f} h on {ln['account']} for {ln['hours']} h -> factor {st['factor'][ln['account']]}")
    save_state(st)


def carry(src, name, dst, step, st):
    """Copy session `name`'s ckpt.pt from account src into dataset dst/fajfuzzy-big-ckpt."""
    c = st.get("carry") or {}
    if c.get("session") == name and c.get("to") == dst:
        if c.get("ready"):
            return True
        settled = time.time() - c.get("ts", 0) >= CARRY_SETTLE
        if settled and "ready" in kaggle(dst, "datasets", "status", f"{dst}/{CARRY}"):
            c["ready"] = True
            save_state(st)
            log(f"carried {name} (step {step}) to {dst}")
            return True
        if time.time() - c.get("ts", 0) < CARRY_WAIT:
            return False
        log(f"dataset {dst}/{CARRY} still not ready after {CARRY_WAIT // 60} min, uploading again")
    d = TMP / "carry"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    kaggle(src, "kernels", "output", f"{src}/{name}", "-p", str(d), "--file-pattern", r"ckpt\.pt$", timeout=2400)
    ck = next(d.rglob("ckpt.pt"), None)
    if not ck:
        log(f"could not download ckpt.pt of {name} from {src}")
        return False
    up = d / "up"
    up.mkdir()
    shutil.move(str(ck), up / "ckpt.pt")
    (up / "dataset-metadata.json").write_text(json.dumps({"title": CARRY, "id": f"{dst}/{CARRY}", "licenses": [{"name": "other"}]}))
    answer = kaggle(dst, "datasets", "status", f"{dst}/{CARRY}").lower()
    exists = not any(x in answer for x in ("403", "404", "not found", "forbidden"))
    out = (kaggle(dst, "datasets", "version", "-p", str(up), "-m", f"{name} step {step}", timeout=3600) if exists
           else kaggle(dst, "datasets", "create", "-p", str(up), timeout=3600))
    shutil.rmtree(d, ignore_errors=True)
    tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    log(f"upload of {name} to {dst}/{CARRY}: {tail}")
    st["carry"] = {"session": name, "to": dst, "step": step, "ts": time.time(), "ready": False}
    save_state(st)
    return False


def push(acct, name, hours, expect, kernels, datasets):
    d = REPO / f"Niepotrzebne/kernel-{name}"
    d.mkdir(parents=True, exist_ok=True)
    code = (REPO / LAUNCHER).read_text(encoding="utf-8")
    for pat, val in {r"^SESSION_HOURS = .*$": f"SESSION_HOURS = {hours}", r"^EXPECT_STEP = .*$": f"EXPECT_STEP = {expect}"}.items():
        code = re.sub(pat, val, code, count=1, flags=re.M)
    (d / "kernel.py").write_text(code, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{acct}/{name}", "title": name, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": datasets, "competition_sources": [],
        "kernel_sources": kernels, "model_sources": [],
    }, indent=2), encoding="utf-8")
    out = kaggle(acct, "kernels", "push", "-p", str(d), timeout=900)
    log(f"push {acct}/{name}: {out.strip().splitlines()[-1] if out.strip() else '(no output)'}")
    if "Maximum batch GPU session count" in out:
        return 3
    return 0 if "successfully pushed" in out else 1


def pick(prev, st):
    """The account for the next session and its hours, or (None, wait seconds)."""
    order = [prev] + [a for a in ORDER if a != prev] if prev else ORDER[:1]
    blocked = {}
    for acct in order:
        prep = status(acct, ACCOUNTS[acct]["prep"])
        if prep != "COMPLETE":
            blocked[acct] = f"data prep {prep}"
            continue
        hours = session_hours(acct, st)
        if not hours:
            blocked[acct] = "no quota"
            continue
        busy = others_running(acct)
        if busy:
            blocked[acct] = "busy: " + ", ".join(busy)
            if acct == prev:
                since = st.setdefault("busy_since", time.time())
                save_state(st)
                if time.time() - since < SWITCH_AFTER:
                    break
            continue
        st.pop("busy_since", None)
        return acct, hours
    if st.get("blocked") != blocked:
        log("waiting: " + "; ".join(f"{a} {why}" for a, why in blocked.items()))
        st["blocked"] = blocked
        save_state(st)
    refresh = [q[2] for q in (quota(a) for a in ORDER) if q]
    if all(why == "no quota" for why in blocked.values()) and refresh:
        return None, max(900, (min(refresh) - datetime.now(timezone.utc)).total_seconds() + 600)
    return None, POLL


def run(st):
    ignore = st.get("ignore", 0)
    while True:
        if STOP_FILE.exists() or DONE_FILE.exists():
            print(f"{'STOP' if STOP_FILE.exists() else 'DONE'} found, exiting", flush=True)
            return
        found = sessions(st)
        calibrate(st, found)
        busy = [f"{PREFIX}{n} {s}" for n, (_, s) in found.items() if s in BUSY]
        if busy:
            print("busy: " + ", ".join(busy), flush=True)
            nap(POLL)
            continue

        done = sorted(n for n, (_, s) in found.items() if s == "COMPLETE")
        last = done[-1] if done else None
        prev = found[last][0] if last else None
        step = last_step(prev, f"{PREFIX}{last}", st) or 0 if last else 0
        if step >= MAX_STEPS:
            DONE_FILE.write_text(f"session: {PREFIX}{last}\naccount: {prev}\n")
            log(f"pretraining finished in {prev}/{PREFIX}{last}, disc.pt is in its output")
            return
        if len(done) >= 2 and step <= (last_step(found[done[-2]][0], f"{PREFIX}{done[-2]}", st) or 0):
            log(f"{PREFIX}{last} did not move training past {PREFIX}{done[-2]}, chain stopped")
            STOP_FILE.touch()
            return

        fails = sum(1 for n, (_, s) in found.items() if n > max(last or 0, ignore) and s in FAILED)
        if fails >= MAX_FAILS:
            log(f"{fails} failed starts in a row, chain stopped")
            STOP_FILE.touch()
            return
        if fails:
            left = RETRY_WAIT - (time.time() - st.get("launch", {}).get("ts", 0))
            if left > 0:
                nap(min(left, WAIT_CAP))
                continue

        acct, hours = pick(prev, st)
        if acct is None:
            nap(min(hours, WAIT_CAP))
            continue
        name = f"{PREFIX}{max(found, default=0) + 1}"
        kernels = [f"{acct}/{ACCOUNTS[acct]['prep']}"]
        if not last:
            datasets = [f"{acct}/{INIT}"]
        elif acct == prev:
            kernels.append(f"{acct}/{PREFIX}{last}")
            datasets = []
        else:
            if not carry(prev, f"{PREFIX}{last}", acct, step, st):
                nap(300)
                continue
            datasets = [f"{acct}/{CARRY}"]
        q = quota(acct)
        rc = push(acct, name, hours, step, kernels, datasets)
        if rc == 0:
            log(f"{name} on {acct}: {hours} h from step {step}")
            st["launch"] = {"name": name, "account": acct, "hours": hours, "used": q[0] if q else None,
                            "ts": time.time(), "measured": False}
            st["where"][name] = acct
            st.pop("blocked", None)
            save_state(st)
            nap(180)
        elif rc == 3:
            nap(600)
        else:
            nap(RETRY_WAIT)


def main():
    global ONCE, STATE, STOP_FILE, DONE_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one turn and exit (cloud workflow)")
    ap.add_argument("--state", type=Path, default=STATE)
    a = ap.parse_args()
    ONCE, STATE = a.once, a.state
    STOP_FILE, DONE_FILE = STATE.parent / "STOP", STATE.parent / "DONE"
    st = load_state()
    try:
        run(st)
    except Pause:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
