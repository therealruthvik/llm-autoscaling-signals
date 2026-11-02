#!/usr/bin/env python3
"""
validation_step12.py - SIMULATED counterpart of the two-replica validation
(validation/two_replica_t4.py on real vLLM, runs/val_kind_step12_v1 on kind):
step12 load, at most two replicas, same policies and seeds.

Usage: python sim/validation_step12.py --out sim/results/validation_step12_v1.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autoscale_sim as S  # noqa: E402

POLICIES = ["queue", "running", "kv-cal", "static-1", "static-2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    sched = sum(float(x.split(":")[1]) for x in S.LOADS["step12"])
    rows = []
    for pol in POLICIES:
        for k in range(args.seeds):
            seed = 1000 + k
            cfg = dict(S.DEFAULT_CFG, max_replicas=2)
            sim = S.Sim(json.loads(json.dumps(S.POLICIES[pol])), cfg,
                        S.make_arrivals(S.LOADS["step12"], seed), seed).run()
            ready = [r.t_ready for r in sim.reps if r.t_created > 0]
            rows.append({"policy": pol, "seed": seed, **S.summarize(sim, window=sched + 30),
                         "ready_lag_s": (min(ready) - 60.0) if ready else None})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
