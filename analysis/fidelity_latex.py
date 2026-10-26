#!/usr/bin/env python3
"""Emit the paper's fidelity table (LaTeX) from fidelity.csv. Usage:
python analysis/fidelity_latex.py runs/emuval_v2/fidelity.csv > paper/fidelity_table.tex"""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
by = {(int(r["rate"]), r["source"]): r for r in rows}
print(r"""\begin{table}[t]
\centering
\caption{Single-replica fidelity: measured vLLM on T4 (mean of three sessions),
emulator on kind, and simulator, at each offered rate. TTFT p50 and TPOT p50 in
seconds and milliseconds; goodput in completed requests/s; KV is peak usage.}
\label{tab:fidelity}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{r|rrr|rrr|rrr}
\toprule
 & \multicolumn{3}{c|}{TTFT p50 (s)} & \multicolumn{3}{c|}{TPOT p50 (ms)} & \multicolumn{3}{c}{peak KV (\%)} \\
rate & T4 & emu & sim & T4 & emu & sim & T4 & emu & sim \\
\midrule""")
for rate in sorted({int(r["rate"]) for r in rows}):
    t, e, s = by[(rate, "T4")], by[(rate, "emulator")], by[(rate, "sim")]
    f = lambda r, k, m=1, d=2: f"{float(r[k]) * m:.{d}f}"
    print(f"{rate} & {f(t,'ttft_p50')} & {f(e,'ttft_p50')} & {f(s,'ttft_p50')} & "
          f"{f(t,'tpot_p50',1000,0)} & {f(e,'tpot_p50',1000,0)} & {f(s,'tpot_p50',1000,0)} & "
          f"{f(t,'max_kv',100,0)} & {f(e,'max_kv',100,0)} & {f(s,'max_kv',100,0)} \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}""")
