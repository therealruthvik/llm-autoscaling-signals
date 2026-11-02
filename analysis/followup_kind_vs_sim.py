#!/usr/bin/env python3
"""
followup_kind_vs_sim.py - follow-up kind campaigns (measured on the emulator)
next to the simulator with the same seeds:
  runs/followup_kind_v1   ingress, running-floor2 x step, ramp   vs sim/results/followup2_v1.csv (base)
  runs/val_kind_step12_v1 two-replica cap, step12               vs sim/results/validation_step12_v1.csv
  runs/trace30_kind_v1    30-minute Azure trace window          vs sim/results/trace30_v1.csv
Kind scale events: changes in the number of non-terminating pods in pods.jsonl.
Kind max pods counts terminating (draining) pods too.

Usage: python analysis/followup_kind_vs_sim.py > analysis/results/followup_kind_vs_sim.txt
"""
import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGNS = [
    ("followup_kind_v1", "sim/results/followup2_v1.csv", lambda r: r.get("study") == "base"),
    ("val_kind_step12_v1", "sim/results/validation_step12_v1.csv", None),
    ("trace30_kind_v1", "sim/results/trace30_v1.csv", None),
]


def kind_trial(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    reqs = [json.loads(line) for line in open(os.path.join(d, "requests.jsonl"))]
    counts = []
    for line in open(os.path.join(d, "pods.jsonl")):
        snap = json.loads(line)
        if "pods" in snap:
            counts.append(sum(1 for p in snap["pods"] if not p["deleting"]))
    events = sum(1 for a, b in zip(counts, counts[1:]) if a != b)
    return cfg, len(reqs), events


def num(rows, k):
    return np.mean([float(r[k]) for r in rows]) if rows else float("nan")


for name, sim_path, sim_filter in CAMPAIGNS:
    kind_rows = list(csv.DictReader(open(os.path.join(ROOT, f"analysis/results/kind_{name}.csv"))))
    sim_rows = [r for r in csv.DictReader(open(os.path.join(ROOT, sim_path)))
                if sim_filter is None or sim_filter(r)]
    print(f"\n== {name} (kind = emulated, sim = SIMULATED, same seeds)")
    print(f"{'load':>7} {'policy':>15} {'n':>2} {'slo kind':>9} {'slo sim':>8} "
          f"{'pods kind':>10} {'pods sim':>9} {'ev kind':>8} {'ev sim':>7}")
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", name, "*", "*"))):
        load, pol = d.split(os.sep)[-2:]
        if load == "config_snapshot":
            continue
        trials = sorted(glob.glob(os.path.join(d, "trial*")))
        info = [kind_trial(t) for t in trials]
        seeds = {str(c["seed"]) for c, _, _ in info}
        k = [r for r in kind_rows if r["load"] == load and r["policy"] == pol]
        s = [r for r in sim_rows if r["policy"] == pol and r["seed"] in seeds
             and r.get("load", load) == load]
        ev_sim = num(s, "scale_events") if s and "scale_events" in s[0] else float("nan")
        print(f"{load:>7} {pol:>15} {len(k):2d} {num(k, 'slo_attainment'):9.3f} "
              f"{num(s, 'slo_attainment'):8.3f} {num(k, 'pod_seconds'):10.0f} "
              f"{num(s, 'pod_seconds'):9.0f} {np.mean([e for _, _, e in info]):8.1f} {ev_sim:7.1f}")
