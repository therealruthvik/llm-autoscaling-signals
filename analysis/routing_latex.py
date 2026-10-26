#!/usr/bin/env python3
"""routing_latex.py - LaTeX table of the SIMULATED routing study.
python analysis/routing_latex.py sim/results/followup_v1.csv > paper/routing_table.tex"""
import csv
import sys

import numpy as np

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["study"] == "routing"]
POLS = ["queue", "running", "kv-cal", "ttft", "static-3"]
RT = ["random", "least", "central"]


def m(pol, load, rt, k):
    return np.mean([float(r[k]) for r in rows
                    if r["policy"] == pol and r["load"] == load and r["value"] == rt])


print(r"""\begin{table}[t]
\centering
\caption{SIMULATED: SLO attainment and TTFT p99 (s) under three routing schemes
(mean of five seeds). random: Kubernetes Service; least: fewest outstanding
requests; central: one shared queue that replicas pull from.}
\label{tab:routing}
\small
\setlength{\tabcolsep}{2.3pt}
\begin{tabular}{lccc|ccc|ccc}
\toprule
 & \multicolumn{3}{c|}{step: SLO} & \multicolumn{3}{c|}{step: TTFT p99} & \multicolumn{3}{c}{ramp: SLO} \\
Policy & rand & least & cent & rand & least & cent & rand & least & cent \\
\midrule""")
for p in POLS:
    cells = [f"{m(p, 'step', rt, 'slo_attainment'):.2f}" for rt in RT]
    cells += [(lambda v: f"{v:.1f}" if v < 10 else f"{v:.0f}")(m(p, 'step', rt, 'ttft_p99')) for rt in RT]
    cells += [f"{m(p, 'ramp', rt, 'slo_attainment'):.2f}" for rt in RT]
    print(f"{p} & " + " & ".join(cells) + r" \\")
print(r"""\bottomrule
\end{tabular}
\end{table}""")
