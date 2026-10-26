#!/usr/bin/env python3
"""
plots.py - paper figures (PDF) from measured and simulated results.

  fig_profile.pdf   measured T4 profile: latency and signals vs offered rate
  fig_main.pdf      kind campaign: SLO attainment and pod-seconds per policy
  fig_timeline.pdf  one step-burst trial: replicas and queue over time
  fig_sweeps.pdf    SIMULATED sensitivity sweeps

Usage: python analysis/plots.py [--campaign runs/main] [--sweep sim/results/sweep_v1.csv]
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
# Validated categorical order (dataviz reference palette, light mode). Three
# slots are below 3:1 contrast, so every line also gets a marker and a direct
# label (secondary encoding).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POLICY_ORDER = ["queue", "running", "kv-cal", "kv-default", "ttft", "qps", "combined", "static-3"]
SWEEP_POLICIES = ["queue", "running", "kv-cal", "ttft", "qps", "combined"]

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# ---------------------------------------------------------------- profile
def fig_profile():
    runs = ["results_run1", "results_run2", "results_run3"]
    rates = [1, 2, 4, 8, 16, 24, 32]
    agg = {k: [] for k in ("ttft_p50", "ttft_p99", "tpot_p50", "running", "waiting", "kv")}
    for rate in rates:
        per = {k: [] for k in agg}
        for run in runs:
            s = {float(r["rate_rps"]): r for r in csv.DictReader(open(os.path.join(ROOT, "data", run, "summary.csv")))}[rate]
            m = [json.loads(l) for l in open(os.path.join(ROOT, "data", run, f"metrics_rate{rate}.jsonl"))]
            per["ttft_p50"].append(float(s["ttft_p50"]))
            per["ttft_p99"].append(float(s["ttft_p99"]))
            per["tpot_p50"].append(float(s["tpot_p50"]) * 1000)
            per["running"].append(max(x["running"] or 0 for x in m))
            per["waiting"].append(max(x["waiting"] or 0 for x in m))
            per["kv"].append(100 * max(x["kv_cache"] or 0 for x in m))
        for k in agg:
            agg[k].append((np.mean(per[k]), np.min(per[k]), np.max(per[k])))

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    panels = [("TTFT (s, log)", ["ttft_p50", "ttft_p99"], ["p50", "p99"], True),
              ("TPOT p50 (ms)", ["tpot_p50"], [None], False),
              ("Requests (peak)", ["running", "waiting"], ["running", "waiting"], False),
              ("KV-cache usage (peak %)", ["kv"], [None], False)]
    for ax, (title, keys, labels, logy) in zip(axes, panels):
        for i, (k, lab) in enumerate(zip(keys, labels)):
            mean = np.array([a[0] for a in agg[k]])
            lo = np.array([a[1] for a in agg[k]])
            hi = np.array([a[2] for a in agg[k]])
            ax.plot(rates, mean, color=SERIES[i], marker=MARKERS[i], markersize=3.5)
            ax.fill_between(rates, lo, hi, color=SERIES[i], alpha=0.15, linewidth=0)
            if lab:
                ax.annotate(lab, (rates[-1], mean[-1]), xytext=(3, 0), textcoords="offset points",
                            color=INK, fontsize=7, va="center")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("offered rate (req/s)")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        if logy:
            ax.set_yscale("log")
    axes[3].axhline(80, color=INK2, linestyle=":", linewidth=1)
    axes[3].annotate("common 80% threshold", (1, 80), xytext=(0, -9), textcoords="offset points",
                     fontsize=6.5, color=INK2)
    axes[3].set_ylim(0, 90)
    fig.tight_layout()
    save(fig, "fig_profile.pdf")


# ---------------------------------------------------------------- campaign
def load_campaign(campaign):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import summarize_trials as T
    rows = [T.trial_row(d) for d in sorted(glob.glob(os.path.join(campaign, "*", "*", "trial*")))
            if os.path.exists(os.path.join(d, "done.json"))]
    return rows, T


def fig_main(campaign):
    rows, T = load_campaign(campaign)
    loads = [l for l in ("step", "ramp") if any(r["load"] == l for r in rows)]
    fig, axes = plt.subplots(len(loads), 2, figsize=(7.2, 1.8 * len(loads)), squeeze=False)
    for li, load in enumerate(loads):
        pols = [p for p in POLICY_ORDER if any(r["load"] == load and r["policy"] == p for r in rows)]
        for ci, (key, label) in enumerate([("slo_attainment", "SLO attainment"),
                                           ("pod_seconds", "pod-seconds (GPU cost)")]):
            ax = axes[li][ci]
            means, errs = zip(*[T.ci95([r[key] for r in rows if r["load"] == load and r["policy"] == p])
                                for p in pols])
            y = np.arange(len(pols))
            ax.barh(y, means, xerr=errs, color=SERIES[0], height=0.62,
                    error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
            ax.set_yticks(y, pols)
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{load}: {label}", fontsize=8, color=INK, loc="left")
            if key == "slo_attainment":
                ax.set_xlim(0, 1)
    fig.tight_layout()
    save(fig, "fig_main.pdf")


def fig_timeline(campaign, load="step", policies=("queue", "kv-default", "static-3"), trial=0):
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.8), sharex=True)
    for i, pol in enumerate(policies):
        d = os.path.join(campaign, load, pol, f"trial{trial}")
        if not os.path.exists(os.path.join(d, "done.json")):
            continue
        t0 = json.load(open(os.path.join(d, "requests.meta.json")))["t_start"]
        polls = [json.loads(l) for l in open(os.path.join(d, "pods.jsonl"))]
        polls = [p for p in polls if "pods" in p]
        t = [p["t"] - t0 for p in polls]
        axes[0].step(t, [len(p["pods"]) for p in polls], where="post", color=SERIES[i], label=pol)
        prom = json.load(open(os.path.join(d, "prometheus.json")))
        series = {}
        for s in prom.get("waiting", []):
            for ts, v in s["values"]:
                series[float(ts)] = series.get(float(ts), 0) + float(v)
        ts = sorted(series)
        axes[1].plot([x - t0 for x in ts], [series[x] for x in ts], color=SERIES[i], label=pol)
    for ax in axes:
        ax.axvspan(60, 180, color=GRID, alpha=0.6, linewidth=0)
    axes[0].set_ylabel("pods")
    axes[1].set_ylabel("waiting requests")
    axes[1].set_xlabel("time since load start (s)")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[0].set_title("step burst (shaded: 20 req/s)", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "fig_timeline.pdf")


# ---------------------------------------------------------------- sweeps
def fig_sweeps(path, load="step"):
    rows = list(csv.DictReader(open(path)))
    sweeps = [("startup", "startup delay (s)", True), ("scrape", "scrape interval (s)", False),
              ("sync", "HPA sync (s)", False), ("stabilization", "stabilization (s)", False),
              ("max_replicas", "max replicas", True), ("queue_threshold", "queue threshold", True)]
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 2.9), sharey="row")
    for col, (sw, label, logx) in enumerate(sweeps):
        sub = [r for r in rows if r["sweep"] == sw and r["load"] == load]
        pols = [p for p in SWEEP_POLICIES if any(r["policy"] == p for r in sub)]
        xs_all = sorted({float(r["value"]) for r in sub})
        for row, key in enumerate(["slo_attainment", "pod_seconds"]):
            ax = axes[row][col]
            for pol in pols:
                i = SWEEP_POLICIES.index(pol)
                xs = sorted({float(r["value"]) for r in sub if r["policy"] == pol})
                ys = [np.mean([float(r[key]) for r in sub
                               if r["policy"] == pol and float(r["value"]) == x]) for x in xs]
                ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], markersize=2.8,
                        linewidth=1.2, label=pol)
            if logx:
                ax.set_xscale("log")
            ax.set_xticks(xs_all, [f"{x:g}" for x in xs_all], fontsize=6, rotation=90)
            ax.minorticks_off()
            if row == 0:
                ax.set_title(label, fontsize=7.5, color=INK, loc="left")
                ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("SLO attainment")
    axes[1][0].set_ylabel("pod-seconds")
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.05), fontsize=7)
    fig.suptitle(f"SIMULATED sensitivity, {load} load (mean of 5 seeds)", fontsize=8, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, f"fig_sweeps_{load}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign")
    ap.add_argument("--sweep")
    args = ap.parse_args()
    fig_profile()
    if args.campaign:
        fig_main(args.campaign)
        fig_timeline(args.campaign)
    if args.sweep:
        for load in ("step", "ramp", "steady"):
            fig_sweeps(args.sweep, load)


if __name__ == "__main__":
    main()
