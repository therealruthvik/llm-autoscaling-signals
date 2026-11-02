#!/usr/bin/env python3
"""
run_campaign.py - run the main policy comparison: every policy x load x trial.

Trials are the outer loop so a partial campaign is still balanced across
policies. Policy order is shuffled per trial (fixed seed) so slow drift on the
laptop does not line up with one policy. Finished trials are skipped, so the
campaign can be stopped and restarted.

Usage: python experiments/run_campaign.py --out runs/main [--trials 5]
"""
import argparse
import os
import random
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = ["queue", "running", "kv-cal", "kv-default", "ttft", "qps",
            "combined", "static-3"]
LOADS = ["step", "ramp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--loads", nargs="+", default=LOADS)
    ap.add_argument("--max-replicas", type=int, default=8)
    args = ap.parse_args()
    # Snapshot the exact emulator and policy configs used by this campaign.
    snap = os.path.join(args.out, "config_snapshot")
    os.makedirs(snap, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "cluster", "vllm-sim.yaml"), snap)
    shutil.copytree(os.path.join(ROOT, "cluster", "policies"),
                    os.path.join(snap, "policies"), dirs_exist_ok=True)
    failures = []
    for trial in range(args.trials):
        order = list(args.policies)
        random.Random(trial).shuffle(order)
        for load in args.loads:
            for policy in order:
                cmd = [sys.executable, os.path.join(ROOT, "experiments", "run_trial.py"),
                       "--policy", policy, "--load", load, "--trial", str(trial),
                       "--out", args.out, "--max-replicas", str(args.max_replicas)]
                r = subprocess.run(cmd)
                if r.returncode != 0:
                    failures.append((load, policy, trial))
                    print(f"!! failed: {load}/{policy}/trial{trial}", flush=True)
    print(f"campaign finished, {len(failures)} failures: {failures}")


if __name__ == "__main__":
    main()
