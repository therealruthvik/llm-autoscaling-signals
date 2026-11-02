#!/usr/bin/env python3
"""
followup2.py - SIMULATED second round of follow-up experiments:

  base      ingress (offered load at the gateway) and the static-floor hybrids
            (queue/running/ingress with minReplicaCount 2) at default settings
  target    ingress target sensitivity
  rate_window  ingress rate window 15 s / 30 s / 1 min
  startup   ingress, running and running-floor2 vs startup delay
  max_seqs  vLLM max_num_seqs 64 / 128 / 256 for the main policies (the latency
            calibration is unchanged; only the admission cap moves)

Same seeds, loads and summary windows as sweep.py and followup.py.
Output: one CSV row per simulation run.
Usage: python sim/followup2.py --out sim/results/followup2_v1.csv [--workers 8]
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

LOADS = ["step", "ramp", "steady"]
BASE = ["ingress", "queue-floor2", "running-floor2", "ingress-floor2",
        "running", "queue", "static-2", "static-3"]
INGRESS_TARGETS = [4.0, 5.0, 6.4, 7.5]
WINDOWS = [15.0, 30.0, 60.0]
STARTUPS = [30.0, 73.0, 170.0, 300.0]
MAX_SEQS = [64, 128, 256]
SEQ_POLICIES = ["queue", "running", "kv-cal", "ingress", "qps", "ttft", "static-3"]


def jobs_for(seeds):
    jobs = []
    for pol, load, s in itertools.product(BASE, LOADS, seeds):
        jobs.append(("base", None, pol, load, s))
    for v, load, s in itertools.product(INGRESS_TARGETS, LOADS, seeds):
        jobs.append(("target", v, "ingress", load, s))
    for v, load, s in itertools.product(WINDOWS, LOADS, seeds):
        jobs.append(("rate_window", v, "ingress", load, s))
    for v, pol, load, s in itertools.product(STARTUPS, ["ingress", "running", "running-floor2"],
                                             LOADS, seeds):
        jobs.append(("startup", v, pol, load, s))
    for v, pol, load, s in itertools.product(MAX_SEQS, SEQ_POLICIES, LOADS, seeds):
        jobs.append(("max_seqs", v, pol, load, s))
    return jobs


def one(job):
    study, value, policy, load, seed = job
    cfg = dict(S.DEFAULT_CFG)
    pol = json.loads(json.dumps(S.POLICIES[policy]))
    if study == "target":
        pol["target"] = value
    elif study in ("rate_window", "startup", "max_seqs"):
        cfg[study] = value
    arr = S.make_arrivals(S.LOADS[load], seed)
    sched = sum(float(x.split(":")[1]) for x in S.LOADS[load])
    sim = S.Sim(pol, cfg, arr, seed).run()
    return {"study": study, "value": value, "policy": policy, "load": load, "seed": seed,
            **S.summarize(sim, window=sched + 30),
            "pod_seconds_tail120": S.summarize(sim, window=sched + 120)["pod_seconds"]}


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
