#!/usr/bin/env python3
"""
kind_vs_sim.py - per-policy mean SLO attainment and pod-seconds, kind (measured
on the emulator) next to the simulator's baseline run with the same seeds.

Usage: python analysis/kind_vs_sim.py analysis/results/kind_trials.csv sim/results/sweep_v1.csv
"""
import csv
import sys

import numpy as np

kind = list(csv.DictReader(open(sys.argv[1])))
sim = [r for r in csv.DictReader(open(sys.argv[2])) if r["sweep"] == "base"]
print(f"{'load':>5} {'policy':>11} {'slo kind':>9} {'slo sim':>8} {'pods kind':>10} {'pods sim':>9} {'pod err':>8}")
for load in ("step", "ramp"):
    for pol in sorted({r["policy"] for r in kind if r["load"] == load}):
        k = [r for r in kind if r["load"] == load and r["policy"] == pol]
        s = [r for r in sim if r["load"] == load and r["policy"] == pol]
        ks, ss = np.mean([float(r["slo_attainment"]) for r in k]), np.mean([float(r["slo_attainment"]) for r in s])
        kp, sp = np.mean([float(r["pod_seconds"]) for r in k]), np.mean([float(r["pod_seconds"]) for r in s])
        print(f"{load:>5} {pol:>11} {ks:>9.3f} {ss:>8.3f} {kp:>10.0f} {sp:>9.0f} {100 * (sp - kp) / kp:>7.1f}%")
