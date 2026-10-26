#!/usr/bin/env python3
"""
autoscale_sim.py - discrete-event simulator of vLLM replicas behind a Kubernetes
Service, autoscaled by an HPA reading Prometheus metrics through KEDA.

Purpose: parameter sweeps (thresholds, scrape interval, HPA sync period,
stabilization window, max replicas, startup delay) that would take days on the
kind cluster. Every result from this file is SIMULATED and must be labelled so.

Replica model (continuous batching, calibrated to the T4 runs in data/):
  - at most MAX_SEQS requests running; the rest wait in a per-replica FIFO
  - one decode iteration produces one token for every running request and
    lasts TPOT(b) for batch size b, interpolated from the measured TPOT p50 vs
    mean running batch at each rate level (mean of the three T4 runs)
  - an admitted request emits its first token after
    (PREFILL_BASE + PREFILL_PER_TOKEN * prompt_tokens) * load_factor(b), then
    joins the decode batch (fit of TTFT vs prompt length at 1 rps; the TPOT fit
    already includes prefill work interleaved with decode, so iterations are
    not lengthened again)
  - KV usage = tokens held by running requests / KV_TOKENS
Cluster model:
  - Service routing (cfg "routing"): "random" = uniform over Ready replicas, per
    request (kube-proxy, the default); "least" = fewest outstanding requests at
    arrival (a least-request gateway); "central" = one shared waiting queue that
    every Ready replica pulls from when it has a free slot (an idealized gateway
    queue; the waiting metric is then the shared queue length)
  - new pods become Ready after startup delay; removed pods leave routing at
    once, drain for PRESTOP seconds, then in-flight requests fail
  - Prometheus scrapes each pod every scrape_interval (random phase)
  - HPA sync every sync_period: external AverageValue target ->
    desired = ceil(total / target); Value target -> ceil(current * value / target);
    10% tolerance; scale-up limited to max(4 pods, 100%) per sync; scale-down
    uses the max desired over the stabilization window; newest pods removed first
"""
import argparse
import heapq
import json
import math
from dataclasses import dataclass, field

import numpy as np

# ---- calibration constants (from data/results_run{1,2,3}; see vllm-sim.yaml)
MAX_SEQS = 256
# (mean running batch, TPOT p50 in s) per rate level 1..32 rps, mean of 3 runs;
# the last point is extrapolated to MAX_SEQS along the final segment.
TPOT_TABLE = [(1.0, 0.0177), (2.8, 0.0177), (5.8, 0.0200), (15.4, 0.0251),
              (71.8, 0.0755), (192.8, 0.1613), (228.2, 0.2113), (233.4, 0.2152),
              (256.0, 0.2322)]
G = 2.5                  # prefill load factor at MAX_SEQS (sim/calibrate.py)
TPOT_SCALE = 0.94        # iteration time = TPOT_SCALE * TPOT(b) (sim/calibrate.py)
PREFILL_BASE = 0.0258       # s, TTFT intercept at 1 rps (fit over 153 requests)
PREFILL_PER_TOKEN = 0.000242  # s per prompt token, same fit (r = 0.97)
KV_TOKENS = 324704       # warm-boot KV capacity on the T4
PRESTOP = 120.0          # s, matches cluster/vllm-sim.yaml
SCHED_DELAY = 3.0        # s, pod scheduling + container start before startup clock


@dataclass
class Req:
    rid: int
    t_arr: float
    pin: int
    pout: int
    t_first: float = None
    t_done: float = None
    gen: int = 0
    failed: bool = False
    replica: int = None


@dataclass
class Replica:
    rid: int
    t_created: float
    t_ready: float
    ready: bool = False
    removed_at: float = None
    dead: bool = False
    waiting: list = field(default_factory=list)
    prefill: list = field(default_factory=list)  # admitted, awaiting prefill
    running: list = field(default_factory=list)
    busy: bool = False
    success: int = 0
    ttft_obs: list = field(default_factory=list)  # (t, ttft) for histogram metric

    def kv(self):
        return sum(r.pin + r.gen for r in self.running + self.prefill) / KV_TOKENS


def load_factor(b, f):
    b = max(1, b)
    return 1 + (f - 1) * (b - 1) / (MAX_SEQS - 1)


_TB = np.array([b for b, _ in TPOT_TABLE])
_TT = np.array([t for _, t in TPOT_TABLE])


def iter_time(b):
    return TPOT_SCALE * float(np.interp(max(1, b), _TB, _TT))


class Sim:
    def __init__(self, policy, cfg, arrivals, seed):
        self.cfg = cfg
        self.policy = policy
        self.rng = np.random.default_rng(seed + 7)
        self.t = 0.0
        self.ev = []
        self.seq = 0
        self.reps = []
        self.reqs = arrivals
        self.desired_hist = []   # (t, desired) for stabilization
        self.scraped = {}        # replica id -> (t, metrics snapshot)
        self.trace = []          # (t, spec, ready, total_pods, metric_value)
        self.spec = policy.get("static", 1)
        self.central = []        # shared waiting queue (routing == "central")
        for _ in range(self.spec):
            self.add_replica(ready_now=True)

    # -- event plumbing
    def push(self, t, kind, data=None):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, data))

    def add_replica(self, ready_now=False):
        rid = len(self.reps)
        t_ready = self.t if ready_now else self.t + SCHED_DELAY + self.cfg["startup"]
        rep = Replica(rid, self.t, t_ready, ready=ready_now)
        self.reps.append(rep)
        if not ready_now:
            self.push(t_ready, "ready", rid)
        # each pod gets its own scrape phase
        self.push(self.t + self.rng.uniform(0, self.cfg["scrape"]), "scrape", rid)

    def live(self):
        return [r for r in self.reps if r.removed_at is None]

    # -- replica engine
    def kick(self, rep):
        if rep.dead:
            return
        pull = self.cfg.get("routing") == "central" and rep.ready and rep.removed_at is None
        while (rep.waiting or (pull and self.central)) and \
                len(rep.running) + len(rep.prefill) < MAX_SEQS:
            r = rep.waiting.pop(0) if rep.waiting else self.central.pop(0)
            r.replica = rep.rid
            rep.prefill.append(r)
            b = len(rep.running) + len(rep.prefill)
            dt = (PREFILL_BASE + PREFILL_PER_TOKEN * r.pin) * load_factor(b, G)
            self.push(self.t + dt, "first", (rep.rid, r))
        if rep.running and not rep.busy:
            rep.busy = True
            self.push(self.t + iter_time(len(rep.running) + len(rep.prefill)),
                      "iter", rep.rid)

    def on_first(self, rid, r):
        rep = self.reps[rid]
        if rep.dead or r.failed:
            return
        rep.prefill.remove(r)
        r.t_first = self.t
        r.gen = 1
        rep.ttft_obs.append((self.t, r.t_first - r.t_arr))
        if r.gen >= r.pout:
            self.finish(rep, r)
        else:
            rep.running.append(r)
        self.kick(rep)

    def finish(self, rep, r):
        r.t_done = self.t
        rep.success += 1

    def on_iter(self, rid):
        rep = self.reps[rid]
        rep.busy = False
        if rep.dead:
            return
        still = []
        for r in rep.running:
            r.gen += 1
            if r.gen >= r.pout:
                self.finish(rep, r)
            else:
                still.append(r)
        rep.running = still
        self.kick(rep)

    # -- autoscaler
    def metric_value(self):
        """Value of the policy signal as KEDA would read it from Prometheus."""
        kind = self.policy["signal"]
        snaps = [self.scraped[r.rid][1] for r in self.live() if r.rid in self.scraped]
        win = self.cfg.get("rate_window", 60.0)
        if kind == "waiting" and self.cfg.get("routing") == "central":
            return sum(s[kind] for s in snaps) + len(self.central)
        if kind in ("waiting", "running", "kv"):
            return sum(s[kind] for s in snaps)
        if kind == "qps":  # completions/s over the last rate_window s, from scraped counters
            tot = 0.0
            for r in self.live():
                hist = [h for h in getattr(r, "succ_hist", []) if h[0] >= self.t - win]
                if len(hist) >= 2:
                    tot += (hist[-1][1] - hist[0][1]) / max(1e-9, hist[-1][0] - hist[0][0])
            return tot
        if kind == "ttft":  # p90 of TTFTs observed in the last rate_window s (as scraped)
            vals = []
            for r in self.live():
                cut = self.scraped.get(r.rid, (0, None))[0]
                vals += [v for (t, v) in r.ttft_obs if self.t - win <= t <= cut]
            return float(np.percentile(vals, 90)) if vals else 0.0
        raise ValueError(kind)

    def hpa_sync(self):
        if "static" in self.policy:
            return
        cur = self.spec
        val = self.metric_value()
        rules = self.policy["triggers"] if "triggers" in self.policy else [self.policy]
        desired = 1
        for rule in rules:
            v = val if rule is self.policy else self.metric_for(rule)
            if rule["type"] == "AverageValue":
                ratio = v / (rule["target"] * cur)
            else:
                ratio = v / rule["target"]
            d = cur if abs(ratio - 1) <= 0.1 else math.ceil(cur * ratio)
            desired = max(desired, d)
        desired = min(max(desired, 1), self.cfg["max_replicas"])
        self.desired_hist.append((self.t, desired))
        win = [d for (t, d) in self.desired_hist if t >= self.t - self.cfg["stabilization"]]
        if desired > cur:
            new = min(desired, cur + max(4, cur))
        else:
            new = min(cur, max(win))
        if new > cur:
            for _ in range(new - cur):
                self.add_replica()
        elif new < cur:
            # newest first; not-ready pods are removed before ready ones
            victims = sorted(self.live(), key=lambda r: (r.ready, -r.t_created))[:cur - new]
            for rep in victims:
                self.remove(rep)
        self.spec = new
        self.trace.append((self.t, new, sum(r.ready for r in self.live()),
                           sum(not r.dead for r in self.reps), val))

    def metric_for(self, rule):
        saved = self.policy
        self.policy = rule
        try:
            return self.metric_value()
        finally:
            self.policy = saved

    def remove(self, rep):
        rep.removed_at = self.t
        rep.ready = False
        self.push(self.t + PRESTOP, "kill", rep.rid)

    def on_kill(self, rid):
        rep = self.reps[rid]
        rep.dead = True
        for r in rep.waiting + rep.prefill + rep.running:
            r.failed = True
            r.t_done = self.t
        rep.waiting, rep.prefill, rep.running = [], [], []

    def on_scrape(self, rid):
        rep = self.reps[rid]
        if rep.dead:
            return
        self.scraped[rid] = (self.t, {"waiting": len(rep.waiting),
                                      "running": len(rep.running) + len(rep.prefill),
                                      "kv": rep.kv()})
        if not hasattr(rep, "succ_hist"):
            rep.succ_hist = []
        rep.succ_hist.append((self.t, rep.success))
        self.push(self.t + self.cfg["scrape"], "scrape", rid)

    def on_arrival(self, i):
        r = self.reqs[i]
        ready = [x for x in self.reps if x.ready and x.removed_at is None]
        if not ready:
            r.failed = True
            r.t_done = self.t
            return
        routing = self.cfg.get("routing", "random")
        if routing == "central":
            self.central.append(r)
            for rep in sorted(ready, key=lambda x: len(x.running) + len(x.prefill)):
                self.kick(rep)
                if not self.central:
                    break
            return
        if routing == "least":
            rep = min(ready, key=lambda x: len(x.waiting) + len(x.prefill) + len(x.running))
        else:
            rep = ready[self.rng.integers(len(ready))]
        r.replica = rep.rid
        rep.waiting.append(r)
        self.kick(rep)

    def run(self):
        for i, r in enumerate(self.reqs):
            self.push(r.t_arr, "arr", i)
        self.push(self.rng.uniform(0, self.cfg["sync"]), "hpa")
        t_last = self.reqs[-1].t_arr if self.reqs else 0
        while self.ev:
            t, _, kind, data = heapq.heappop(self.ev)
            self.t = t
            if kind == "arr":
                self.on_arrival(data)
            elif kind == "iter":
                self.on_iter(data)
            elif kind == "first":
                self.on_first(*data)
            elif kind == "ready":
                rep = self.reps[data]
                if rep.removed_at is None:
                    rep.ready = True
                    self.kick(rep)
            elif kind == "kill":
                self.on_kill(data)
            elif kind == "scrape":
                self.on_scrape(data)
                if t > t_last + 900 and all(r.t_done is not None for r in self.reqs):
                    break
            elif kind == "hpa":
                self.hpa_sync()
                if not (t > t_last + 900 and all(r.t_done is not None for r in self.reqs)):
                    self.push(t + self.cfg["sync"], "hpa")
        return self


# ------------------------------------------------------------------ workloads
def make_arrivals(schedule, seed, in_mean=256, out_mean=128, max_in=1024, max_out=512):
    """Same arrival and length process as cluster/loadgen.py."""
    rng = np.random.default_rng(seed)
    reqs, t = [], 0.0
    for seg in schedule:
        rate, dur = (float(x) for x in seg.split(":"))
        end = t + dur
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= end:
                t = end
                break
            pin = int(np.clip(rng.lognormal(np.log(in_mean), 0.6), 8, max_in))
            pout = int(np.clip(rng.lognormal(np.log(out_mean), 0.6), 8, max_out))
            rng.choice(25, size=pin)  # keep the RNG stream aligned with loadgen's prompt draw
            reqs.append(Req(len(reqs), t, pin, pout))
    return reqs


# ------------------------------------------------------------------ policies
POLICIES = {
    "queue": {"signal": "waiting", "type": "AverageValue", "target": 5},
    "running": {"signal": "running", "type": "AverageValue", "target": 50},
    "kv-cal": {"signal": "kv", "type": "AverageValue", "target": 0.06},
    "kv-default": {"signal": "kv", "type": "AverageValue", "target": 0.8},
    "ttft": {"signal": "ttft", "type": "Value", "target": 1.0},
    "qps": {"signal": "qps", "type": "AverageValue", "target": 6.4},
}
POLICIES["combined"] = {"signal": "waiting", "type": "AverageValue", "target": 5,
                        "triggers": [POLICIES["queue"] | {}, POLICIES["kv-default"] | {}]}
POLICIES["combined-cal"] = {"signal": "waiting", "type": "AverageValue", "target": 5,
                            "triggers": [POLICIES["queue"] | {}, POLICIES["kv-cal"] | {}]}
for n in (1, 2, 3, 4):
    POLICIES[f"static-{n}"] = {"static": n}

LOADS = {
    "step": ["2:60", "20:120", "2:180"],
    "ramp": [f"{r}:40" for r in (2, 5, 9, 14, 18, 20, 18, 14, 9, 5, 2)],
    "steady": ["12:360"],
}
DEFAULT_CFG = {"startup": 73.0, "scrape": 15.0, "sync": 15.0,
               "stabilization": 60.0, "max_replicas": 8,
               "routing": "random", "rate_window": 60.0}


def summarize(sim, slo_ttft=1.0, slo_tpot=0.15, window=None):
    reqs = sim.reqs
    ok = [r for r in reqs if not r.failed and r.t_first is not None]
    ttft = np.array([r.t_first - r.t_arr for r in ok])
    tpot = np.array([(r.t_done - r.t_first) / (r.pout - 1) if r.pout > 1 else 0 for r in ok])
    meet = sum(1 for a, b in zip(ttft, tpot) if a <= slo_ttft and b <= slo_tpot)
    end = window or (reqs[-1].t_arr if reqs else 0)
    pod_s = 0.0
    for rep in sim.reps:
        t1 = min(end, rep.removed_at + PRESTOP if rep.removed_at is not None else end)
        pod_s += max(0.0, t1 - rep.t_created)
    return {
        "n": len(reqs), "failed": sum(r.failed for r in reqs),
        "slo_attainment": meet / len(reqs) if reqs else None,
        "ttft_p50": float(np.percentile(ttft, 50)) if len(ttft) else None,
        "ttft_p99": float(np.percentile(ttft, 99)) if len(ttft) else None,
        "tpot_p50": float(np.percentile(tpot, 50)) if len(tpot) else None,
        "pod_seconds": pod_s,
        "max_replicas_used": max([s for (_, s, *_rest) in sim.trace] or [sim.spec]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="queue", choices=sorted(POLICIES))
    ap.add_argument("--load", default="step", choices=sorted(LOADS))
    ap.add_argument("--seed", type=int, default=1000)
    for k, v in DEFAULT_CFG.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--queue-threshold", type=float, default=None)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULT_CFG}
    pol = json.loads(json.dumps(POLICIES[args.policy]))
    if args.queue_threshold is not None and pol.get("signal") == "waiting":
        pol["target"] = args.queue_threshold
        for t in pol.get("triggers", []):
            if t["signal"] == "waiting":
                t["target"] = args.queue_threshold
    sim = Sim(pol, cfg, make_arrivals(LOADS[args.load], args.seed), args.seed).run()
    print(json.dumps({"policy": args.policy, "load": args.load, "cfg": cfg,
                      **summarize(sim)}, indent=2))


if __name__ == "__main__":
    main()
