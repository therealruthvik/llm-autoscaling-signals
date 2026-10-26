#!/usr/bin/env python3
"""
sweep.py - SIMULATED sensitivity sweeps (RQ2, RQ3). One parameter is varied at a
time around the baseline config; every policy x load x seed is run per value.

Output: one CSV row per simulation run.

Usage: python sim/sweep.py --out sim/results/sweep.csv [--workers 2] [--seeds 5]
"""
import argparse
import csv
import itertools
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autoscale_sim as S  # noqa: E402

BASE_POLICIES = ["queue", "running", "kv-cal", "kv-default", "ttft", "qps", "combined",
                 "static-1", "static-2", "static-3", "static-4"]
SWEEP_POLICIES = ["queue", "running", "kv-cal", "ttft", "qps", "combined"]
SWEEPS = {
    "queue_threshold": [1, 2, 5, 10, 25, 50],
    "scrape": [5.0, 15.0, 30.0, 60.0],
    "sync": [5.0, 15.0, 30.0],
    "stabilization": [0.0, 60.0, 300.0],
    "max_replicas": [2, 3, 4, 8, 16, 32],
    "startup": [30.0, 73.0, 100.0, 170.0, 210.0, 300.0, 600.0],
}
LOADS = ["step", "ramp", "steady"]


def one(job):
    sweep, value, policy, load, seed = job
    cfg = dict(S.DEFAULT_CFG)
    pol = json.loads(json.dumps(S.POLICIES[policy]))
    if sweep == "queue_threshold":
        if pol.get("signal") != "waiting":
            return None
        pol["target"] = value
        for t in pol.get("triggers", []):
            if t["signal"] == "waiting":
                t["target"] = value
    elif sweep != "base":
        cfg[sweep] = value
    arr = S.make_arrivals(S.LOADS[load], seed)
    sched = sum(float(x.split(":")[1]) for x in S.LOADS[load])
    sim = S.Sim(pol, cfg, arr, seed).run()
    # pod_seconds: schedule + 30 s, the window kind trials are polled over
    # (analysis/summarize_trials.py). pod_seconds_tail120 also counts the
    # scale-down tail, which is where stabilization shows up.
    long = S.summarize(sim, window=sched + 120)["pod_seconds"]
    return {"sweep": sweep, "value": value, "policy": policy, "load": load, "seed": seed,
            **S.summarize(sim, window=sched + 30), "pod_seconds_tail120": long}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = [1000 + k for k in range(args.seeds)]  # same seeds as the kind trials
    jobs = [("base", None, p, l, s) for p, l, s in itertools.product(BASE_POLICIES, LOADS, seeds)]
    for sweep, values in SWEEPS.items():
        pols = ["queue", "combined"] if sweep == "queue_threshold" else SWEEP_POLICIES
        jobs += [(sweep, v, p, l, s) for v, p, l, s in itertools.product(values, pols, LOADS, seeds)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"{len(jobs)} simulation runs", flush=True)
    with Pool(args.workers) as pool, open(args.out, "w", newline="") as f:
        w = None
        for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if row is None:
                continue
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done", args.out)


if __name__ == "__main__":
    main()
