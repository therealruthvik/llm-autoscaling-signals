#!/usr/bin/env python3
"""
trace_study.py - SIMULATED 24 h replay of one weekday (2024-05-13, Monday) of
the Azure LLM conversation trace (sim/trace.py), thinned so the busiest minute
offers `peak` requests/s. Compares reactive policies, static provisioning and
static-floor hybrids over a full diurnal cycle, where autoscaling can save cost.

Each seed thins the trace differently and draws different request lengths.
Pod-seconds cover the whole day (24 h); cost is also reported as mean replicas.
Per-hour attainment is recorded so the diurnal pattern can be plotted.

Output: one CSV row per run.
Rerunning with the same --out resumes: finished (peak, policy, seed) rows are kept.

Usage: python sim/trace_study.py --out sim/results/trace_v1.csv [--workers 8]
"""
import argparse
import csv
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autoscale_sim as S  # noqa: E402
import trace as T  # noqa: E402

DAY = "2024-05-13"
HOURS = 24.0
PEAKS = [20.0, 40.0]   # busiest-minute rate after thinning, requests/s
REACTIVE = ["queue", "running", "kv-cal", "ttft", "qps", "ingress",
            "queue-floor2", "running-floor2", "ingress-floor2"]
STATIC = {20.0: [1, 2, 3, 4], 40.0: [2, 3, 4, 5, 6]}


def jobs_for(seeds):
    jobs = []
    for peak, s in itertools.product(PEAKS, seeds):
        for pol in REACTIVE:
            jobs.append((peak, pol, s))
        for n in STATIC[peak]:
            jobs.append((peak, f"static-{n}", s))
    return jobs


def one(job):
    peak, policy, seed = job
    cfg = dict(S.DEFAULT_CFG)
    if peak > 20:
        cfg["max_replicas"] = 16
    pol = ({"static": int(policy.split("-")[1])} if policy.startswith("static-")
           else json.loads(json.dumps(S.POLICIES[policy])))
    reqs, k = T.make_arrivals(DAY, peak, seed, hours=HOURS)
    sim = S.Sim(pol, cfg, reqs, seed).run()
    end = HOURS * 3600
    row = {"day": DAY, "peak": peak, "thin": k, "policy": policy, "seed": seed,
           "max_replicas": cfg["max_replicas"], **S.summarize(sim, window=end)}
    row["mean_replicas"] = row["pod_seconds"] / end
    # hourly attainment (by arrival hour)
    hours = np.array([r.t_arr // 3600 for r in reqs]).astype(int)
    meet = np.array([(not r.failed and r.t_first is not None
                      and r.t_first - r.t_arr <= 1.0
                      and (r.pout <= 1 or (r.t_done - r.t_first) / (r.pout - 1) <= 0.15))
                     for r in reqs])
    row["hourly_attainment"] = json.dumps(
        [round(float(meet[hours == h].mean()), 4) if (hours == h).any() else None
         for h in range(int(HOURS))])
    row["scale_events"] = sum(1 for a, b in zip(sim.trace, sim.trace[1:]) if a[1] != b[1])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    jobs = jobs_for([1000 + k for k in range(args.seeds)])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # resume: keep finished rows, run only the missing jobs
    done, fields = set(), None
    if os.path.exists(args.out):
        with open(args.out) as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames
            done = {(float(r["peak"]), r["policy"], int(r["seed"])) for r in rd}
    jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} simulation runs ({len(done)} already done)", flush=True)
    with Pool(args.workers) as pool, open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields) if fields else None
        for i, row in enumerate(pool.imap_unordered(one, jobs)):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)
            f.flush()
            print(f"{i + 1}/{len(jobs)} {row['peak']} {row['policy']} {row['seed']}", flush=True)
    print("done", args.out)


if __name__ == "__main__":
    main()
