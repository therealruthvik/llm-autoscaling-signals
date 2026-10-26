#!/usr/bin/env python3
"""followup_summary.py - tables for sim/results/followup_v1.csv (SIMULATED)."""
import csv
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "sim/results/followup_v1.csv"
rows = list(csv.DictReader(open(path)))
g = defaultdict(list)
for r in rows:
    g[(r["study"], r["value"], r["policy"], r["load"])].append(r)


def m(rs, k):
    return np.mean([float(r[k]) for r in rs])


for study in ["routing", "target", "rate_window", "static"]:
    print(f"\n== {study} (SIMULATED, mean of 5 seeds): slo / slo@ttft1.5 / slo@ttft2 / slo@ttft5 / pod-s(+30) / pod-s(+120) / max pods")
    keys = sorted({k for k in g if k[0] == study}, key=lambda k: (k[3], k[2], k[1]))
    for k in keys:
        rs = g[k]
        print(f"{k[3]:>6} {k[2]:>12} {k[1]:>8}  {m(rs,'slo_attainment'):.3f} {m(rs,'slo_ttft1.5'):.3f} "
              f"{m(rs,'slo_ttft2'):.3f} {m(rs,'slo_ttft5'):.3f}  {m(rs,'pod_seconds'):7.0f} "
              f"{m(rs,'pod_seconds_tail120'):7.0f}  {m(rs,'max_replicas_used'):.1f}")
