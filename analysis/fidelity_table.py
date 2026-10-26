#!/usr/bin/env python3
"""
fidelity_table.py - single-replica behavior at each offered rate:
measured vLLM on T4 (mean of 3 runs) vs emulator on kind vs simulator.

Usage: python analysis/fidelity_table.py runs/emuval_v1 [--csv out.csv]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import autoscale_sim as S  # noqa: E402

RATES = [1, 2, 4, 8, 16, 24, 32]
RUNS = ["results_run1", "results_run2", "results_run3"]


def stats(ttft, tpot, n_ok, wall, running=None, waiting=None, kv=None):
    p = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")
    return {"ttft_p50": p(ttft, 50), "ttft_p99": p(ttft, 99), "tpot_p50": p(tpot, 50),
            "goodput": n_ok / wall if wall else float("nan"),
            "max_running": running, "max_waiting": waiting, "max_kv": kv}


def measured(rate):
    rows = []
    for run in RUNS:
        reqs = [json.loads(l) for l in open(os.path.join(ROOT, "data", run, f"requests_rate{rate}.jsonl"))]
        ok = [r for r in reqs if r.get("ok")]
        m = [json.loads(l) for l in open(os.path.join(ROOT, "data", run, f"metrics_rate{rate}.jsonl"))]
        wall = max(r["t_send"] + r["e2e"] for r in ok) - min(r["t_send"] for r in reqs)
        rows.append(stats(np.array([r["ttft"] for r in ok]),
                          np.array([r["tpot"] for r in ok if r.get("tpot")]), len(ok), wall,
                          max(x["running"] or 0 for x in m), max(x["waiting"] or 0 for x in m),
                          max(x["kv_cache"] or 0 for x in m)))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def emulator(d, rate):
    reqs = [json.loads(l) for l in open(os.path.join(d, f"requests_rate{rate}.jsonl"))]
    ok = [r for r in reqs if not r["error"] and r["ttft"] is not None]
    m = [json.loads(l) for l in open(os.path.join(d, f"requests_rate{rate}.metrics.jsonl"))]
    wall = max(r["t_send"] + r["e2e"] for r in ok) - min(r["t_send"] for r in reqs)
    return stats(np.array([r["ttft"] for r in ok]), np.array([r["tpot"] for r in ok if r["tpot"]]),
                 len(ok), wall, max((x.get("running") or 0) for x in m),
                 max((x.get("waiting") or 0) for x in m), max((x.get("kv_cache") or 0) for x in m))


def simulated(rate, seed=42):
    sim = S.Sim({"static": 1}, dict(S.DEFAULT_CFG), S.make_arrivals([f"{rate}:60"], seed), seed)
    peak = {"running": 0, "waiting": 0, "kv": 0.0}
    orig = sim.on_scrape

    def scrape(rid):  # track peaks at 1 Hz-ish resolution via scrape hook
        orig(rid)
        rep = sim.reps[rid]
        peak["running"] = max(peak["running"], len(rep.running) + len(rep.prefill))
        peak["waiting"] = max(peak["waiting"], len(rep.waiting))
        peak["kv"] = max(peak["kv"], rep.kv())
    sim.on_scrape = scrape
    sim.cfg["scrape"] = 1.0
    sim.run()
    ok = [r for r in sim.reqs if r.t_first and r.t_done]
    wall = max(r.t_done for r in ok) - min(r.t_arr for r in sim.reqs)
    return stats(np.array([r.t_first - r.t_arr for r in ok]),
                 np.array([(r.t_done - r.t_first) / (r.pout - 1) for r in ok if r.pout > 1]),
                 len(ok), wall, peak["running"], peak["waiting"], peak["kv"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emu_dir")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = []
    for rate in RATES:
        srcs = {"T4": measured(rate), "sim": simulated(rate)}
        if os.path.exists(os.path.join(args.emu_dir, f"requests_rate{rate}.metrics.jsonl")) and \
                os.path.getsize(os.path.join(args.emu_dir, f"requests_rate{rate}.jsonl")) > 0:
            srcs["emulator"] = emulator(args.emu_dir, rate)
        for src, st in srcs.items():
            rows.append({"rate": rate, "source": src, **st})
    keys = ["ttft_p50", "ttft_p99", "tpot_p50", "goodput", "max_running", "max_waiting", "max_kv"]
    print(f"{'rate':>4} {'source':>8} " + " ".join(f"{k:>11}" for k in keys))
    for r in rows:
        print(f"{r['rate']:>4} {r['source']:>8} " + " ".join(f"{r[k]:>11.3f}" for k in keys))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["rate", "source"] + keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
