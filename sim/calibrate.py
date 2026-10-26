#!/usr/bin/env python3
"""
calibrate.py - fit the simulator's prefill load factor G and iteration-time scale so a
single simulated replica reproduces the measured T4 rate sweep.

Target: per-rate TTFT p50 and TPOT p50 and goodput, averaged over
data/results_run{1,2,3}, all rates 1-32.
Loss: mean squared log error over the three quantities.
"""
import csv
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autoscale_sim as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATES = [1, 2, 4, 8, 16, 24, 32]


def measured():
    out = {}
    for run in ("results_run1", "results_run2", "results_run3"):
        for r in csv.DictReader(open(os.path.join(ROOT, "data", run, "summary.csv"))):
            k = float(r["rate_rps"])
            out.setdefault(k, []).append((float(r["ttft_p50"]), float(r["tpot_p50"]),
                                          float(r["goodput_req_s"])))
    return {k: np.mean(v, axis=0) for k, v in out.items()}


def simulate(rate, seed=42):
    sim = S.Sim({"static": 1}, dict(S.DEFAULT_CFG),
                S.make_arrivals([f"{rate}:60"], seed), seed).run()
    ok = [r for r in sim.reqs if r.t_first and r.t_done]
    ttft = np.array([r.t_first - r.t_arr for r in ok])
    tpot = np.array([(r.t_done - r.t_first) / (r.pout - 1) for r in ok if r.pout > 1])
    wall = max(r.t_done for r in ok) - min(r.t_arr for r in sim.reqs)
    return np.median(ttft), np.median(tpot), len(ok) / wall


def loss(G, scale, meas):
    S.G, S.TPOT_SCALE = G, scale
    err = []
    for rate in RATES:
        sim = simulate(rate)
        err += list((np.log(sim) - np.log(meas[rate])) ** 2)
    return float(np.mean(err))


def saturated_rate(seed=42):
    """Completions/s while fully saturated at 32 rps offered (same window as
    used on the measured runs: completions between 60 s and end-30 s)."""
    sim = S.Sim({"static": 1}, dict(S.DEFAULT_CFG), S.make_arrivals(["32:60"], seed), seed).run()
    e = np.array([r.t_done for r in sim.reqs if r.t_done])
    mid = (e > 60) & (e < e.max() - 30)
    return mid.sum() / ((e.max() - 30) - 60)


if __name__ == "__main__":
    meas = measured()
    # 1) iteration-time scale: match measured saturated completion rate at
    #    32 rps (7.83-7.88/s across the three runs; mean 7.85)
    S.G = 2.5
    scale = min(np.arange(0.80, 1.01, 0.02), key=lambda sc: abs(
        (setattr(S, "TPOT_SCALE", sc) or saturated_rate()) - 7.85))
    S.TPOT_SCALE = scale
    # 2) prefill load factor: minimize log error on TTFT/TPOT/goodput, rates 1-32
    G = min(np.arange(1, 8.5, 0.5), key=lambda g: loss(g, scale, meas))
    S.G = G
    print(f"TPOT_SCALE={scale:.2f} (saturated {saturated_rate():.2f}/s vs measured 7.85/s)  G={G}")
    print("rate | measured ttft/tpot/goodput | simulated")
    for rate in RATES:
        m = meas[float(rate)]
        s = simulate(rate)
        print(f"{rate:4} | {m[0]:7.3f} {m[1]:6.3f} {m[2]:5.2f} | {s[0]:7.3f} {s[1]:6.3f} {s[2]:5.2f}")
