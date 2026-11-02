#!/usr/bin/env python3
"""
val_t4x2_compare.py - step12 on two real vLLM replicas (MEASURED, Kaggle T4 x2,
validation/two_replica_t4.py) next to the emulated kind cluster and the
simulator with the same seeds:
  runs/val_t4x2_v1         measured, real vLLM 0.32.0, HPA-equivalent loop
  runs/val_kind_step12_v1  emulated (kind + KEDA + llm-d-inference-sim)
  sim/results/validation_step12_v1.csv  SIMULATED

Also reports replica startup times on the T4s (create -> /health OK, from
events.jsonl) and the requests that failed on the real testbed, with their
error strings.

Usage: python analysis/val_t4x2_compare.py > analysis/results/val_t4x2_compare.txt
"""
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from summarize_trials import trial_row  # noqa: E402

REAL = os.path.join(ROOT, "runs/val_t4x2_v1")
KIND = os.path.join(ROOT, "runs/val_kind_step12_v1")
SIM = os.path.join(ROOT, "sim/results/validation_step12_v1.csv")
KEYS = ["slo_attainment", "ttft_p99", "pod_seconds", "ready_lag_s", "failed"]


def campaign(root):
    out = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "step12", "*", "trial*"))):
        if os.path.exists(os.path.join(d, "done.json")):
            out[os.path.basename(os.path.dirname(d))].append(trial_row(d))
    return out


def mean(rows, k):
    v = [float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")]
    v = [x for x in v if not np.isnan(x)]
    return np.mean(v) if v else float("nan")


def main():
    real, kind = campaign(REAL), campaign(KIND)
    sim = collections.defaultdict(list)
    for r in csv.DictReader(open(SIM)):
        sim[r["policy"]].append(r)

    print("== step12, mean over 3 seeds (real = MEASURED T4 x2, kind = emulated, sim = SIMULATED)")
    print(f"{'policy':>9}  {'n':>2}  " + "  ".join(f"{k + ' ' + s:>15}" for k in KEYS[:4]
                                                  for s in ("real", "kind", "sim")))
    for p in sorted(real):
        cells = []
        for k in KEYS[:4]:
            cells += [mean(real[p], k), mean(kind[p], k), mean(sim[p], k)]
        print(f"{p:>9}  {len(real[p]):>2}  " + "  ".join(f"{c:15.3f}" for c in cells))

    print("\n== rank by SLO attainment")
    for name, src in (("real", real), ("kind", kind), ("sim", sim)):
        order = sorted(src, key=lambda p: -mean(src[p], "slo_attainment"))
        print(f"{name:>5}: " + " > ".join(f"{p} ({mean(src[p], 'slo_attainment'):.3f})" for p in order))

    print("\n== per-trial SLO attainment, real (MEASURED)")
    for p in sorted(real):
        print(f"{p:>9}: " + ", ".join(f"{float(r['slo_attainment']):.3f}" for r in real[p]))

    print("\n== replica startup on the T4s (create -> /health OK), MEASURED")
    first, scaled = [], []
    for ev in sorted(glob.glob(os.path.join(REAL, "step12", "*", "trial*", "events.jsonl"))):
        rows = [json.loads(l) for l in open(ev)]
        t_end = next((r["t"] for r in rows if r["event"] == "trial_end"), None)
        ready = [r for r in rows if r["event"] == "ready"]
        n_static = 2 if "/static-2/" in ev else 1
        first += [r["startup_s"] for r in ready[:n_static]]
        scaled += [r["startup_s"] for r in ready[n_static:] if t_end is None or r["t"] < t_end]
    for name, v in (("initial (before load)", first), ("autoscaler scale-up", scaled)):
        if v:
            print(f"  {name:>22}: n={len(v)} median {np.median(v):.1f} s, "
                  f"min {min(v):.1f}, max {max(v):.1f}")
    warm = os.path.join(REAL, "warmup_events.jsonl")
    if os.path.exists(warm):
        w = [json.loads(l) for l in open(warm) if '"ready"' in l]
        print("  warm-up (first boot per GPU): " +
              ", ".join(f"GPU {r['gpu']} {r['startup_s']:.1f} s" for r in w))

    print("\n== failed requests on the real testbed (MEASURED)")
    total, errs = 0, collections.Counter()
    for f in sorted(glob.glob(os.path.join(REAL, "step12", "*", "trial*", "requests.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            total += 1
            if r.get("error"):
                errs[str(r["error"])[:100]] += 1
    print(f"  {sum(errs.values())} of {total} requests failed")
    for e, c in errs.most_common():
        print(f"  {c:4d}  {e}")


if __name__ == "__main__":
    main()
