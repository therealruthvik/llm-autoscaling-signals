#!/usr/bin/env python3
"""
slo_sensitivity.py - MEASURED (kind campaign) SLO attainment under relaxed TTFT
and TPOT limits, and the split of misses into TTFT-only, TPOT-only and both.

Usage: python analysis/slo_sensitivity.py runs/main
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

LIMITS = [(1.0, 0.15), (2.0, 0.15), (5.0, 0.15), (1.0, 0.25), (5.0, 0.25)]
camp = sys.argv[1] if len(sys.argv) > 1 else "runs/main"
acc = defaultdict(list)
for d in sorted(glob.glob(os.path.join(camp, "*", "*", "trial*"))):
    if not os.path.exists(os.path.join(d, "done.json")):
        continue
    cfg = json.load(open(os.path.join(d, "config.json")))
    reqs = [json.loads(l) for l in open(os.path.join(d, "requests.jsonl"))]
    n = len(reqs)
    ok = [r for r in reqs if not r["error"] and r["ttft"] is not None]
    tt = np.array([r["ttft"] for r in ok])
    tp = np.array([r["tpot"] if r["tpot"] is not None else 0.0 for r in ok])
    row = [np.sum((tt <= a) & (tp <= b)) / n for a, b in LIMITS]
    row += [np.sum((tt > 1) & (tp <= .15)) / n, np.sum((tt <= 1) & (tp > .15)) / n,
            np.sum((tt > 1) & (tp > .15)) / n]
    acc[(cfg["load"], cfg["policy"])].append(row)
hdr = " ".join(f"ttft{a:g}/tpot{b:g}" for a, b in LIMITS)
print(f"MEASURED kind campaign, mean over trials\n{'load':>5} {'policy':>11}  {hdr}  | miss: ttft-only tpot-only both")
for (load, pol), rows in sorted(acc.items()):
    m = np.mean(rows, 0)
    print(f"{load:>5} {pol:>11}  " + "  ".join(f"{x:13.3f}" for x in m[:len(LIMITS)])
          + "  | " + " ".join(f"{x:.3f}" for x in m[len(LIMITS):]))
