#!/usr/bin/env python3
"""
followup.py - SIMULATED follow-up experiments requested in review:

  routing        random (kube-proxy) vs least-request vs central queue, all policies
  target         per-policy target sensitivity (running, kv-cal, qps, ttft)
  rate_window    1 m vs 30 s vs 15 s rate window for the qps and ttft signals
  combined_cal   queue OR KV at the calibrated 0.06 target (vs 0.8)
  static         static-1..4 frontier
  slo            SLO attainment at TTFT limits 1, 1.5, 2, 5 s (same runs)

Every run also records SLO attainment at the relaxed TTFT limits, so the slo
study needs no extra runs.

Output: one CSV row per simulation run.
Usage: python sim/followup.py --out sim/results/followup_v1.csv [--workers 8]
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

REACTIVE = ["queue", "running", "kv-cal", "kv-default", "ttft", "qps", "combined"]
LOADS = ["step", "ramp", "steady"]
TTFT_LIMITS = [1.0, 1.5, 2.0, 5.0]
TARGETS = {
    "running": [30, 50, 70, 100],
    "kv-cal": [0.04, 0.06, 0.1, 0.2],
    "qps": [4.0, 5.0, 6.4, 7.5],
    "ttft": [0.75, 1.0, 1.5, 2.0],
}


def jobs_for(seeds):
    jobs = []
    for pol, load, s, rt in itertools.product(REACTIVE + ["combined-cal", "static-3"], LOADS,
                                              seeds, ["random", "least", "central"]):
        jobs.append(("routing", rt, pol, load, s))
    for pol, vals in TARGETS.items():
        for v, load, s in itertools.product(vals, LOADS, seeds):
            jobs.append(("target", v, pol, load, s))
    for pol, v, load, s in itertools.product(["qps", "ttft"], [15.0, 30.0, 60.0], LOADS, seeds):
        jobs.append(("rate_window", v, pol, load, s))
    for n, load, s in itertools.product([1, 2, 3, 4], LOADS, seeds):
        jobs.append(("static", n, f"static-{n}", load, s))
    return jobs


def one(job):
    study, value, policy, load, seed = job
    cfg = dict(S.DEFAULT_CFG)
    pol = json.loads(json.dumps(S.POLICIES[policy]))
    if study == "routing":
        cfg["routing"] = value
    elif study == "target":
        pol["target"] = value
    elif study == "rate_window":
        cfg["rate_window"] = value
    arr = S.make_arrivals(S.LOADS[load], seed)
    sched = sum(float(x.split(":")[1]) for x in S.LOADS[load])
    sim = S.Sim(pol, cfg, arr, seed).run()
    row = {"study": study, "value": value, "policy": policy, "load": load, "seed": seed,
           **S.summarize(sim, window=sched + 30),
           "pod_seconds_tail120": S.summarize(sim, window=sched + 120)["pod_seconds"]}
    for lim in TTFT_LIMITS[1:]:
        row[f"slo_ttft{lim:g}"] = S.summarize(sim, slo_ttft=lim)["slo_attainment"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    jobs = jobs_for([1000 + k for k in range(args.seeds)])  # same seeds as the kind trials
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"{len(jobs)} simulation runs", flush=True)
    with Pool(args.workers) as pool, open(args.out, "w", newline="") as f:
        w = None
        for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done", args.out)


if __name__ == "__main__":
    main()
