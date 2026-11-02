#!/usr/bin/env python3
"""
summarize_trials.py - one row per kind trial, plus per-policy means with 95% CIs.

Per trial (runs/<campaign>/<load>/<policy>/trial<k>/):
  slo_attainment  share of ALL sent requests with TTFT <= SLO_TTFT and
                  TPOT <= SLO_TPOT (failed requests count as misses)
  ttft_p50/p99, tpot_p50 over successful requests
  failed          requests that errored (e.g. dropped at scale-down)
  pod_seconds     integral of pods that exist (starting, ready, or draining:
                  each holds a GPU) over schedule + 30 s, from pods.jsonl
  scale_lag_s     burst start -> first extra pod created (step load only)
  ready_lag_s     burst start -> first extra pod Ready (step load only)

Usage: python analysis/summarize_trials.py runs/main [--csv results.csv]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy import stats

SLO_TTFT = 1.0
SLO_TPOT = 0.15
WINDOW_TAIL = 30.0  # pods are polled until ~30 s after the schedule; sim uses the same tail
BURST_START = {"step": 60.0, "ramp": None, "steady": None, "step12": 60.0}


def schedule_len(schedule):
    return sum(float(s.split(":")[1]) for s in schedule)


def trial_row(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    reqs = [json.loads(l) for l in open(os.path.join(d, "requests.jsonl"))]
    meta = json.load(open(os.path.join(d, "requests.meta.json")))
    t0 = meta["t_start"]
    ok = [r for r in reqs if not r["error"] and r["ttft"] is not None]
    ttft = np.array([r["ttft"] for r in ok])
    tpot = np.array([r["tpot"] if r["tpot"] is not None else 0.0 for r in ok])
    meet = int(np.sum((ttft <= SLO_TTFT) & (tpot <= SLO_TPOT)))

    # pod-seconds over [t0, t0 + schedule + WINDOW_TAIL]
    window_end = t0 + schedule_len(cfg["schedule"]) + WINDOW_TAIL
    polls = [json.loads(l) for l in open(os.path.join(d, "pods.jsonl"))]
    polls = [p for p in polls if "pods" in p]
    pod_s, prev_t, prev_n = 0.0, None, None
    first_extra, first_extra_ready = None, None
    for p in polls:
        t, n = p["t"], len(p["pods"])
        if prev_t is not None:
            a, b = max(prev_t, t0), min(t, window_end)
            if b > a:
                pod_s += prev_n * (b - a)
        prev_t, prev_n = t, n
        if t >= t0:
            if first_extra is None and n > 1:
                first_extra = t - t0
            if first_extra_ready is None and sum(x["ready"] for x in p["pods"]) > 1:
                first_extra_ready = t - t0
    burst = BURST_START.get(cfg["load"])
    lag = lambda x: (x - burst) if (burst is not None and x is not None) else None
    max_pods = max((len(p["pods"]) for p in polls if p["t"] >= t0), default=None)
    return {
        "load": cfg["load"], "policy": cfg["policy"], "trial": cfg["trial"],
        "n": len(reqs), "failed": len(reqs) - len(ok),
        "slo_attainment": meet / len(reqs),
        "ttft_p50": float(np.percentile(ttft, 50)), "ttft_p99": float(np.percentile(ttft, 99)),
        "tpot_p50": float(np.percentile(tpot, 50)),
        "pod_seconds": pod_s, "max_pods": max_pods,
        "scale_lag_s": lag(first_extra), "ready_lag_s": lag(first_extra_ready),
    }


def ci95(x):
    x = np.array([v for v in x if v is not None], dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(args.campaign, "*", "*", "trial*"))):
        if os.path.exists(os.path.join(d, "done.json")):
            rows.append(trial_row(d))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    keys = ["slo_attainment", "ttft_p99", "failed", "pod_seconds", "max_pods", "ready_lag_s"]
    for load in sorted({r["load"] for r in rows}):
        print(f"\n== {load}  (mean ± 95% CI over trials)")
        print(f"{'policy':>15} {'n':>2} " + " ".join(f"{k:>20}" for k in keys))
        for pol in sorted({r["policy"] for r in rows if r["load"] == load}):
            rs = [r for r in rows if r["load"] == load and r["policy"] == pol]
            cells = []
            for k in keys:
                m, h = ci95([r[k] for r in rs])
                cells.append(f"{m:>11.3f} ± {h:<6.3f}")
            print(f"{pol:>15} {len(rs):>2} " + " ".join(cells))


if __name__ == "__main__":
    main()
