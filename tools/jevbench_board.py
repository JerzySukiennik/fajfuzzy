"""Fetch the open-weights JevBench board (benchmarkheaven.com/jev-models) and keep every row's
Intelligence, Calibration, Capability, cost and latency as JSON, read from the server-rendered
ranking (the published aggregate JSON is an older release).

    python tools/jevbench_board.py --out board-live.json
"""

import argparse
import json
import re
import urllib.request
from html.parser import HTMLParser

URL = "https://benchmarkheaven.com/jev-models"


class Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cur, self.depth = {}, None, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.cur is None and tag == "li" and "data-bh-jev14-capability-row" in a:
            key = a["data-bh-jev14-capability-row"]
            cat = re.search(r"--jev-a-([\w-]+)", a.get("style") or "")
            self.cur = {"key": key, "cap": float(a["data-bh-jev14-capability-value"]),
                        "usd": float(a["data-bh-jev14-cost"]), "cat": cat.group(1) if cat else None,
                        "text": [], "title": None}
            self.depth = 1
            return
        if self.cur is not None:
            if tag == "li":
                self.depth += 1
            if self.cur["title"] is None and "data-bh-jev-capability-name-cell" in a:
                self.cur["title"] = a.get("title")

    def handle_endtag(self, tag):
        if self.cur is not None and tag == "li":
            self.depth -= 1
            if self.depth == 0:
                if self.cur["key"] not in self.rows:
                    self.rows[self.cur["key"]] = self.cur
                self.cur = None

    def handle_data(self, data):
        if self.cur is not None:
            self.cur["text"].append(data)


def num(s, pat):
    m = re.search(pat, s)
    return float(m.group(1)) if m else None


def rows(html):
    p = Rows()
    p.feed(html)
    out = []
    for r in p.rows.values():
        t = "|".join(x.strip() for x in r["text"] if x.strip())
        out.append({
            "key": r["key"], "title": r["title"], "cat": r["cat"], "rank": t.split("|")[0],
            "intelligence": num(t, r"Intelligence\|?([\d.]+)"),
            "calibration": num(t, r"Calibration\|?([\d.]+)"),
            "capability": r["cap"],
            "cost_score": num(t, r"Cost Score\|?([\d.]+)"),
            "usd_per_1000": r["usd"],
            "p50_measured_s": num(t, r"measured p50 ([\d.]+) s"),
            "outside": (re.search(r"Outside: ([^|]+)", t) or [None, None])[1],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--html", help="read a saved copy of the page instead of fetching it")
    a = ap.parse_args()
    if a.html:
        html = open(a.html, encoding="utf-8").read()
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    out = rows(html)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(out)} systems, {sum(r['rank'] != '–' for r in out)} ranked")


if __name__ == "__main__":
    main()
