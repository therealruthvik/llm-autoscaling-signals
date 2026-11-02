#!/usr/bin/env python3
"""
trace.py - arrival times from the Azure LLM inference trace 2024 (conversation,
one week, CC-BY; Stojkovic et al., DynamoLLM, HPCA 2025) for SIMULATED replay.

The raw CSV (1.1 GB, not in git) is in data/traces/azure2024/; its URL and
SHA-256 are in data/PROVENANCE.md. `extract` writes one day's arrival offsets
(seconds since midnight UTC) to a small .npy that is kept in git.

Replay keeps the trace's arrival timestamps and thins them: each request is kept
with probability k, chosen so that the busiest minute of the day offers
`peak_rps` requests/s. Thinning keeps the trace's burst and diurnal structure.
Prompt and output lengths are NOT taken from the trace (context mean 1,631
tokens, beyond the calibrated range and the 2,048-token max_model_len); they are
drawn from the same lognormal mix as the T4 profile (mean 256 / 128 tokens).

The kind trace trials replay a 30-minute window (TRACE30) through
cluster/loadgen.py --arrivals; `arrivals` writes those files, and `read_arrivals`
loads the same file into the simulator so both see identical requests.

Usage: python sim/trace.py extract 2024-05-13
       python sim/trace.py arrivals 2024-05-13 --seeds 1000 1001 1002
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "data", "traces", "azure2024")
RAW = os.path.join(TRACE_DIR, "AzureLLMInferenceTrace_conv_1week.csv")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autoscale_sim as S  # noqa: E402


def day_path(day):
    return os.path.join(TRACE_DIR, f"conv_{day}.npy")


def extract(day):
    """Arrival offsets in seconds since 00:00 UTC of `day`, sorted."""
    out = []
    with open(RAW) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ts = row[0]
            if not ts.startswith(day):
                continue
            h, m, sec = ts[11:].split("+")[0].split(":")
            out.append(int(h) * 3600 + int(m) * 60 + float(sec))
    a = np.sort(np.array(out))
    np.save(day_path(day), a)
    return a


def load_day(day):
    return np.load(day_path(day))


def thin_factor(times, peak_rps):
    per_min = np.bincount((times // 60).astype(int))
    return min(1.0, peak_rps / (per_min.max() / 60.0))


def make_arrivals(day, peak_rps, seed, start_h=0.0, hours=24.0,
                  in_mean=256, out_mean=128, max_in=1024, max_out=512):
    """Thinned trace arrivals in [start_h, start_h + hours), times from 0."""
    times = load_day(day)
    k = thin_factor(times, peak_rps)  # from the whole day, so windows share one scale
    lo, hi = start_h * 3600, (start_h + hours) * 3600
    times = times[(times >= lo) & (times < hi)] - lo
    rng = np.random.default_rng(seed)
    keep = rng.random(len(times)) < k
    reqs = []
    for t in times[keep]:
        pin = int(np.clip(rng.lognormal(np.log(in_mean), 0.6), 8, max_in))
        pout = int(np.clip(rng.lognormal(np.log(out_mean), 0.6), 8, max_out))
        reqs.append(S.Req(len(reqs), float(t), pin, pout))
    return reqs, k


# 30-minute window with the widest per-minute rate range of the day at peak 20
# (15:50-16:20 UTC on 2024-05-13: 11.6-18.1 req/s per minute, mean 15.0).
TRACE30 = {"day": "2024-05-13", "start_h": 15 + 50 / 60, "hours": 0.5, "peak": 20.0}


def arrivals_path(seed):
    return os.path.join(TRACE_DIR, "trace30", f"seed{seed}.arrivals.jsonl")


def write_arrivals(seed):
    reqs, _ = make_arrivals(TRACE30["day"], TRACE30["peak"], seed,
                            start_h=TRACE30["start_h"], hours=TRACE30["hours"])
    os.makedirs(os.path.dirname(arrivals_path(seed)), exist_ok=True)
    with open(arrivals_path(seed), "w") as f:
        for r in reqs:
            f.write(json.dumps({"t": round(r.t_arr, 6), "pin": r.pin, "pout": r.pout}) + "\n")
    return len(reqs)


def read_arrivals(path):
    return [S.Req(i, a["t"], a["pin"], a["pout"])
            for i, a in enumerate(map(json.loads, open(path)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "arrivals"])
    ap.add_argument("day")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    args = ap.parse_args()
    if args.cmd == "extract":
        a = extract(args.day)
        print(f"{args.day}: {len(a)} arrivals -> {day_path(args.day)}")
    else:
        for seed in args.seeds:
            print(f"seed {seed}: {write_arrivals(seed)} requests -> {arrivals_path(seed)}")


if __name__ == "__main__":
    main()
