#!/usr/bin/env python3
"""
paired_stats.py - paired differences between policies on the kind trials.

Trial k uses the same arrival seed for every policy, so policy A minus policy B
is computed per trial and summarized as mean difference with a 95% CI
(Student's t over the paired differences), a paired t-test p value and the
exact two-sided Wilcoxon signed-rank p value. With five pairs the smallest
possible Wilcoxon p value is 0.0625, so the t-based CI is the primary summary.

Usage: python analysis/paired_stats.py analysis/results/kind_trials.csv \
           [more.csv ...] > analysis/results/paired_stats.txt
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

PAIRS = [
    ("running", "queue"), ("kv-cal", "queue"), ("running", "kv-cal"),
    ("combined", "queue"), ("ttft", "queue"), ("qps", "queue"),
    ("ingress", "qps"), ("ingress", "running"), ("running-floor2", "running"),
    ("running-floor2", "static-3"),
]
METRICS = ["slo_attainment", "pod_seconds"]


def main(paths):
    data = defaultdict(dict)  # (load, policy) -> trial -> row
    for p in paths:
        for r in csv.DictReader(open(p)):
            data[(r["load"], r["policy"])][(p, int(r["trial"]))] = r
    loads = sorted({k[0] for k in data})
    print("Paired differences A - B over trials with the same seed (kind cluster, emulated).")
    print("mean diff, 95% CI (t), paired t p, Wilcoxon p (exact, two-sided)\n")
    for load in loads:
        print(f"== {load}")
        for a, b in PAIRS:
            A, B = data.get((load, a)), data.get((load, b))
            if not A or not B:
                continue
            # pair on the seed (trial index); campaigns share seeds 1000 + k
            ta = {k[1]: v for k, v in A.items()}
            tb = {k[1]: v for k, v in B.items()}
            common = sorted(set(ta) & set(tb))
            if len(common) < 2:
                continue
            for m in METRICS:
                d = np.array([float(ta[t][m]) - float(tb[t][m]) for t in common])
                mean = d.mean()
                if np.all(d == d[0]):
                    ci, pt, pw = 0.0, float("nan"), float("nan")
                else:
                    ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    pt = stats.ttest_1samp(d, 0).pvalue
                    pw = stats.wilcoxon(d, method="exact").pvalue if np.any(d != 0) else float("nan")
                print(f"  {a:>15} - {b:<9} {m:<15} n={len(d)}  {mean:+10.3f} "
                      f"[{mean - ci:+.3f}, {mean + ci:+.3f}]  t p={pt:.3g}  W p={pw:.3g}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
