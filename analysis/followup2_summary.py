#!/usr/bin/env python3
"""followup2_summary.py - tables for sim/results/followup2_v1.csv (SIMULATED)."""
import csv
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "sim/results/followup2_v1.csv"
rows = list(csv.DictReader(open(path)))
g = defaultdict(list)
for r in rows:
    g[(r["study"], r["value"], r["policy"], r["load"])].append(r)


def m(rs, k):
    return np.mean([float(r[k]) for r in rs])


def num(v):
    try:
        return float(v)
    except ValueError:
        return -1.0


for study in ["base", "target", "rate_window", "startup", "max_seqs"]:
    print(f"\n== {study} (SIMULATED, mean of 5 seeds): slo / ttft p99 / pod-s(+30) / pod-s(+120) / max pods")
    keys = sorted({k for k in g if k[0] == study}, key=lambda k: (k[3], k[2], num(k[1])))
    for k in keys:
        rs = g[k]
        print(f"{k[3]:>6} {k[2]:>15} {k[1]:>6}  {m(rs,'slo_attainment'):.3f} {m(rs,'ttft_p99'):7.1f} "
              f"{m(rs,'pod_seconds'):7.0f} {m(rs,'pod_seconds_tail120'):7.0f}  {m(rs,'max_replicas_used'):.1f}")
