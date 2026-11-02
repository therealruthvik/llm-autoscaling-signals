#!/usr/bin/env python3
"""trace_mechanism.py - why saturation signals oscillate on the trace.

1. MEASURED (T4): mean running requests per offered req/s at each rate level.
   Running grows faster than load, so it is not proportional to load.
2. SIMULATED: the 30-minute trace window (sim/trace.py TRACE30, seed 1000) per
   policy: scale-ups, scale-downs, and the share of HPA syncs at which the
   ready replica count is below the count the offered load needs
   (ceil(arrivals over the last minute / 6.4 req/s, the 80% target)).
   The per-sync trace is printed for the first 15 minutes of running.

Output: analysis/results/trace_mechanism.txt (stdout).
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import autoscale_sim as S  # noqa: E402
import trace as T  # noqa: E402

TARGET = 6.4  # 80% of measured capacity, req/s per replica

print("== MEASURED (T4): mean running requests at each offered rate, per session")
for rate in (1, 2, 4, 8):
    vals = []
    for run in (1, 2, 3):
        rs = [json.loads(line) for line in
              open(os.path.join(ROOT, f"data/results_run{run}/metrics_rate{rate}.jsonl"))]
        vals.append(np.mean([r["running"] for r in rs]))
    print(f"rate {rate:2d}: running {' '.join(f'{v:5.1f}' for v in vals)}"
          f"   per req/s {np.mean(vals) / rate:4.1f}")

print("\n== SIMULATED: TRACE30 window, seed 1000")
print(f"{'policy':>15} {'ups':>4} {'downs':>5} {'ready below need':>16}")
detail = None
for pol in ("queue", "running", "kv-cal", "ingress", "running-floor2"):
    reqs = T.read_arrivals(T.arrivals_path(1000))
    arr = np.array([r.t_arr for r in reqs])
    sim = S.Sim(json.loads(json.dumps(S.POLICIES[pol])), dict(S.DEFAULT_CFG), reqs, 1000).run()
    tr = [x for x in sim.trace if x[0] <= T.TRACE30["hours"] * 3600]
    ups = sum(1 for a, b in zip(tr, tr[1:]) if b[1] > a[1])
    downs = sum(1 for a, b in zip(tr, tr[1:]) if b[1] < a[1])
    need = [math.ceil(((arr > t - 60) & (arr <= t)).sum() / 60 / TARGET) for t, *_ in tr]
    below = np.mean([x[2] < n for x, n in zip(tr, need) if x[0] >= 60])
    print(f"{pol:>15} {ups:4d} {downs:5d} {below:16.2f}")
    if pol == "running":
        detail = list(zip(tr, need))

print("\n== SIMULATED: running, first 15 min (every second sync)")
print(f"{'t':>5} {'need':>4} {'spec':>4} {'ready':>5} {'signal':>7}")
for (t, spec, ready, _, m), n in detail[::2]:
    if t > 900:
        break
    print(f"{t:5.0f} {n:4d} {spec:4d} {ready:5d} {m:7.0f}")
