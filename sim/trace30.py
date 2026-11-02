#!/usr/bin/env python3
"""
trace30.py - SIMULATED counterpart of the kind trace trials: the same 30-minute
Azure trace window and the same arrival files (sim/trace.py TRACE30), same
policies and seeds, summarized over the window plus 30 s as on kind.

Usage: python sim/trace30.py --out sim/results/trace30_v1.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autoscale_sim as S  # noqa: E402
import trace as T  # noqa: E402

POLICIES = ["queue", "running", "ingress", "running-floor2", "static-2", "static-3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    args = ap.parse_args()
    rows = []
    for pol in POLICIES:
        for seed in args.seeds:
            reqs = T.read_arrivals(T.arrivals_path(seed))
            sim = S.Sim(json.loads(json.dumps(S.POLICIES[pol])), dict(S.DEFAULT_CFG),
                        reqs, seed).run()
            rows.append({"policy": pol, "seed": seed,
                         **S.summarize(sim, window=T.TRACE30["hours"] * 3600 + 30),
                         "scale_events": sum(1 for a, b in zip(sim.trace, sim.trace[1:])
                                             if a[1] != b[1])})
            print(rows[-1]["policy"], seed, round(rows[-1]["slo_attainment"], 3),
                  round(rows[-1]["pod_seconds"]), rows[-1]["scale_events"], flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
