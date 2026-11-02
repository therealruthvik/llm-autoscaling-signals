#!/usr/bin/env python3
"""Summarize the measured long-prompt T4 profile (validation/RUNBOOK.md, section A).

For each rate level: request outcomes, TTFT p99, peak waiting/running/KV usage
from the 1 Hz metrics trace, and the first time (seconds from the first trace
sample of the level) that KV usage reaches 0.8 and that waiting reaches 5.

Usage: python analysis/longctx_summary.py [data/results_longctx_run1] \
           > analysis/results/longctx_summary.txt
"""
import csv
import json
import os
import sys

KV_T = 0.8
QUEUE_T = 5


def first_cross(rows, key, thr):
    for x in rows:
        if x[key] >= thr:
            return x["t"] - rows[0]["t"], x
    return None, None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/results_longctx_run1"
    print(f"MEASURED (T4), {d}")
    # summary.csv writes 1.0, the trace files are named metrics_rate1.jsonl
    summary = {f"{float(r['rate_rps']):g}": r
               for r in csv.DictReader(open(os.path.join(d, "summary.csv")))}
    print("rate  ok/sent  fail  ttft_p99_s  goodput  peak_wait  peak_run  peak_kv  mean_kv"
          "  t_kv>=0.8  t_wait>=5")
    for rate, s in summary.items():
        rows = [json.loads(l) for l in open(os.path.join(d, f"metrics_rate{rate}.jsonl"))]
        kv = [x["kv_cache"] for x in rows]
        t_kv, x_kv = first_cross(rows, "kv_cache", KV_T)
        t_q, x_q = first_cross(rows, "waiting", QUEUE_T)
        fmt = lambda t: "-" if t is None else f"{t:.0f}s"
        print(f"{rate:>4}  {s['ok']:>3}/{s['sent']:<4} {s['fail']:>4}  {float(s['ttft_p99']):10.1f}"
              f"  {float(s['goodput_req_s']):7.2f}  {max(x['waiting'] for x in rows):9.0f}"
              f"  {max(x['running'] for x in rows):8.0f}  {max(kv):7.3f}  {sum(kv) / len(kv):7.3f}"
              f"  {fmt(t_kv):>9}  {fmt(t_q):>9}")
        if x_kv:
            print(f"      at KV>={KV_T}: waiting {x_kv['waiting']:.0f}, running {x_kv['running']:.0f}")
        if x_q:
            print(f"      at waiting>={QUEUE_T}: KV {x_q['kv_cache']:.3f}")
    pre = max(json.loads(l).get("preemptions", 0)
              for r in summary for l in open(os.path.join(d, f"metrics_rate{r}.jsonl")))
    print(f"max preemptions counter: {pre:.0f}")
    print("coldstart:", json.load(open(os.path.join(d, "coldstart.json"))))


if __name__ == "__main__":
    main()
