#!/usr/bin/env python3
"""
burst_bound.py - an upper bound on SLO attainment under the step load that
holds for any scaling signal, next to the attainment measured on the kind
cluster (runs/main via analysis/results/kind_trials.csv).

Until the first added replica is ready, every request goes to the original
replica. It admits up to CAP requests into the running batch and completes at
most MU requests/s (measured saturated capacity). With LAMBDA requests/s
arriving, the batch is full after t_fill = CAP / (LAMBDA - MU) s at the latest
(fewer completions only fill it sooner). From then on arrivals wait in a FIFO
queue that grows at LAMBDA - MU per s; once it holds MU requests (one second of
service), every later arrival waits more than 1 s and misses the TTFT limit.
So arrivals in [t_fill + MU / (LAMBDA - MU), ready_lag] all miss, and

    attainment <= 1 - LAMBDA * (ready_lag - t_miss) / n

with ready_lag capped at the burst length (no added replica in time).

Usage: python analysis/burst_bound.py [analysis/results/kind_trials.csv] \
           > analysis/results/burst_bound.txt
"""
import collections
import csv
import sys

import numpy as np

LAMBDA = 20.0   # step burst rate, requests/s
BURST = 120.0   # burst length, s
MU = 7.85       # measured saturated single-replica completion rate, requests/s
CAP = 256       # max_num_seqs
SLO_TTFT = 1.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "analysis/results/kind_trials.csv"
    t_fill = CAP / (LAMBDA - MU)
    t_miss = t_fill + MU * SLO_TTFT / (LAMBDA - MU)
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["load"] != "step" or r["policy"].startswith("static"):
            continue
        lag = float(r["ready_lag_s"]) if r["ready_lag_s"] not in ("", "nan") else BURST
        lag = min(lag, BURST)
        bound = 1 - LAMBDA * max(0.0, lag - t_miss) / int(r["n"])
        rows[r["policy"]].append((lag, bound, float(r["slo_attainment"])))
    print("step load, kind cluster (emulated), 5 trials per policy")
    print(f"t_fill <= {t_fill:.1f} s, arrivals miss from {t_miss:.1f} s after burst start "
          f"until an added replica is ready")
    print(f"{'policy':>11}  {'ready_lag_s':>11}  {'bound':>6}  {'measured':>8}  {'bound-measured':>14}")
    for p in sorted(rows, key=lambda p: -np.mean([x[2] for x in rows[p]])):
        lag, b, m = (np.mean([x[i] for x in rows[p]]) for i in range(3))
        ok = all(x[2] <= x[1] + 1e-9 for x in rows[p])
        print(f"{p:>11}  {lag:11.1f}  {b:6.3f}  {m:8.3f}  {b - m:14.3f}"
              + ("" if ok else "  BOUND VIOLATED"))


if __name__ == "__main__":
    main()
