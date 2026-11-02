#!/usr/bin/env python3
"""trace_summary.py - table for sim/results/trace_v1.csv (SIMULATED 24 h replay
of the Azure LLM conversation trace, 2024-05-13). Mean over seeds; cost in mean
replicas over the day (pod-seconds / 86,400 s) and relative to the smallest
static size that meets the SLO for at least 99% of requests."""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "sim/results/trace_v1.csv"
rows = list(csv.DictReader(open(path)))
g = defaultdict(list)
for r in rows:
    g[(float(r["peak"]), r["policy"])].append(r)


def m(rs, k):
    return float(np.mean([float(r[k]) for r in rs]))


day_times = np.load("data/traces/azure2024/conv_2024-05-13.npy")
per_hour = np.bincount((day_times // 3600).astype(int), minlength=24)[:24] / 3600
per_min = np.bincount((day_times // 60).astype(int)) / 60

for peak in sorted({k[0] for k in g}):
    k_thin = m(g[next(k for k in g if k[0] == peak)], "thin")
    print(f"\n== offered load at peak {peak:g}: hourly mean {k_thin * per_hour.min():.1f}-"
          f"{k_thin * per_hour.max():.1f} req/s (trough hour {per_hour.argmin()}, "
          f"busiest hour {per_hour.argmax()} UTC), busiest minute {k_thin * per_min.max():.1f}")
    keys = [k for k in g if k[0] == peak]
    rs0 = g[keys[0]]
    print(f"\n== peak {peak:g} req/s (thinning {m(rs0, 'thin'):.3f}, "
          f"{m(rs0, 'n') / 86400:.1f} req/s mean, {len(rs0)} seeds), SIMULATED")
    statics = sorted((k for k in keys if k[1].startswith("static-")),
                     key=lambda k: int(k[1].split("-")[1]))
    ref = next((k for k in statics if m(g[k], "slo_attainment") >= 0.99), None)
    ref_cost = m(g[ref], "mean_replicas") if ref else float("nan")
    print(f"reference: {ref[1] if ref else 'none'} ({ref_cost:.2f} replicas)")
    print(f"{'policy':>15} {'slo':>6} {'min-hour':>8} {'ttft_p99':>9} {'replicas':>8} "
          f"{'vs ref':>6} {'max':>4} {'scale ev':>8}")
    order = sorted(keys, key=lambda k: (not k[1].startswith("static-"), k[1]))
    for k in order:
        rs = g[k]
        hourly = np.array([[v if v is not None else np.nan for v in json.loads(r["hourly_attainment"])]
                           for r in rs])
        worst = float(np.nanmin(np.nanmean(hourly, axis=0)))
        print(f"{k[1]:>15} {m(rs, 'slo_attainment'):6.3f} {worst:8.3f} {m(rs, 'ttft_p99'):9.1f} "
              f"{m(rs, 'mean_replicas'):8.2f} {m(rs, 'mean_replicas') / ref_cost:6.2f} "
              f"{m(rs, 'max_replicas_used'):4.0f} {m(rs, 'scale_events'):8.0f}")
