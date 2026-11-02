#!/usr/bin/env python3
"""
two_replica_t4.py - real-vLLM validation of the autoscaling mechanisms on a
Kaggle "GPU T4 x2" session (two real replicas, one per GPU).

There is no Kubernetes on Kaggle, so this script reproduces the parts of the
control loop that the paper's conclusions depend on, around two real vLLM
servers:
  router   per-request uniform random choice over Ready replicas (like the
           kind Service with one connection per request), streaming passthrough
  scraper  reads each Ready replica's /metrics every SCRAPE s (random phase),
           like Prometheus
  scaler   every SYNC s applies the HPA AverageValue rule with 10% tolerance,
           max 2 replicas, and a 60 s scale-down stabilization window
  replica  scale-up starts vLLM on the idle GPU (a warm restart: model and
           compile caches are already on disk); it joins the router when
           /health answers. Scale-down removes it from the router, lets it
           drain for PRESTOP s, then stops it.

Output per trial, in the same layout as experiments/run_trial.py so that
analysis/summarize_trials.py reads it unchanged:
  <out>/<load>/<policy>/trial<k>/{config.json, requests.jsonl,
  requests.meta.json, pods.jsonl, events.jsonl, done.json, server_*.log}

Usage on Kaggle (see validation/RUNBOOK.md):
  python validation/two_replica_t4.py --warmup
  python validation/two_replica_t4.py --campaign --out /kaggle/working/val_t4x2 --trials 3
"""
import argparse
import asyncio
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

import aiohttp
from aiohttp import web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROUTER_PORT = 8000
SCRAPE = 15.0
SYNC = 15.0
STABILIZATION = 60.0
PRESTOP = 120.0
MAX_REPLICAS = 2  # two GPUs
TAIL = 30.0       # seconds polled after the schedule, as on kind

# Same signals and targets as cluster/policies (80% of measured T4 capacity).
POLICIES = {
    "queue": ("vllm:num_requests_waiting", 5.0),
    "running": ("vllm:num_requests_running", 50.0),
    "kv-cal": ("vllm:kv_cache_usage_perc", 0.06),
    "static-1": None,
    "static-2": None,
}
# Burst sized so that two replicas suffice (12 req/s = 1.5x one replica).
LOADS = {"step12": ["2:60", "12:120", "2:180"]}


class Replica:
    def __init__(self, gpu):
        self.gpu = gpu
        self.port = 8001 + gpu
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = None
        self.logf = None
        self.state = "off"      # off | starting | ready | draining
        self.t_created = None

    def start(self, log_path):
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", MODEL, "--port", str(self.port), "--dtype", "half",
               "--max-model-len", "2048", "--gpu-memory-utilization", "0.85"]
        if os.environ.get("FAKE_VLLM"):  # harness self-test without a GPU
            cmd = [sys.executable, os.path.join(ROOT, "validation", "fake_vllm.py"),
                   "--port", str(self.port)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu))
        self.logf = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT,
                                     env=env, preexec_fn=os.setsid)
        self.state = "starting"
        self.t_created = time.time()

    def stop(self):
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        if self.logf:
            self.logf.close()
        self.proc, self.logf, self.state, self.t_created = None, None, "off", None


def parse_metric(text, name):
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            total += float(line.rsplit(" ", 1)[1])
            found = True
    return total if found else None


class Cluster:
    def __init__(self, policy, trial_dir):
        self.policy = policy
        self.dir = trial_dir
        self.reps = [Replica(0), Replica(1)]
        self.spec = 1
        self.scraped = {}           # gpu -> metric value
        self.desired_hist = []      # (t, desired)
        self.events = open(os.path.join(trial_dir, "events.jsonl"), "w")
        self.session = None

    def log(self, **kw):
        self.events.write(json.dumps({"t": time.time(), **kw}) + "\n")
        self.events.flush()

    def ready(self):
        return [r for r in self.reps if r.state == "ready"]

    # ---- replica lifecycle
    async def wait_health(self, rep, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if rep.state != "starting":  # removed while starting
                return
            if rep.proc is None or rep.proc.poll() is not None:
                raise RuntimeError(f"vLLM on GPU {rep.gpu} exited; see server log")
            try:
                async with self.session.get(f"{rep.url}/health",
                                            timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200 and rep.state == "starting":
                        rep.state = "ready"
                        self.log(event="ready", gpu=rep.gpu,
                                 startup_s=time.time() - rep.t_created)
                        return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"GPU {rep.gpu} not ready")

    async def scale_up(self, rep):
        rep.start(os.path.join(self.dir, f"server_gpu{rep.gpu}_{int(time.time())}.log"))
        self.log(event="create", gpu=rep.gpu)
        await self.wait_health(rep)

    async def scale_down(self, rep):
        rep.state = "draining"
        self.log(event="remove", gpu=rep.gpu)
        await asyncio.sleep(PRESTOP)
        rep.stop()
        self.log(event="killed", gpu=rep.gpu)

    # ---- control loop
    async def scrape_loop(self, rep, stop):
        await asyncio.sleep(random.uniform(0, SCRAPE))
        name = POLICIES[self.policy][0]
        while not stop.is_set():
            if rep.state == "ready":
                try:
                    async with self.session.get(f"{rep.url}/metrics") as r:
                        v = parse_metric(await r.text(), name)
                    self.scraped[rep.gpu] = v or 0.0
                    self.log(event="scrape", gpu=rep.gpu, value=v)
                except Exception as e:
                    self.log(event="scrape_error", gpu=rep.gpu, error=repr(e))
            else:
                self.scraped.pop(rep.gpu, None)
            await asyncio.sleep(SCRAPE)

    async def hpa_loop(self, stop):
        _, target = POLICIES[self.policy]
        await asyncio.sleep(random.uniform(0, SYNC))
        while not stop.is_set():
            cur = self.spec
            total = sum(v for g, v in self.scraped.items()
                        if self.reps[g].state == "ready")
            ratio = total / (target * cur)
            desired = cur if abs(ratio - 1) <= 0.1 else math.ceil(cur * ratio)
            desired = min(max(desired, 1), MAX_REPLICAS)
            now = time.time()
            self.desired_hist.append((now, desired))
            window = [d for t, d in self.desired_hist if t >= now - STABILIZATION]
            new = desired if desired > cur else min(cur, max(window))
            self.log(event="sync", total=total, cur=cur, desired=desired, new=new)
            if new > cur:
                idle = [r for r in self.reps if r.state == "off"]
                if idle:
                    self.spec = new
                    asyncio.create_task(self.scale_up(idle[0]))
                else:
                    self.log(event="blocked", reason="no idle GPU (still draining)")
            elif new < cur:
                self.spec = new
                # newest first; not-ready pods before ready ones
                live = [r for r in self.reps if r.state in ("starting", "ready")]
                victim = sorted(live, key=lambda r: (r.state == "ready", -r.t_created))[0]
                asyncio.create_task(self.scale_down(victim))
            await asyncio.sleep(SYNC)

    async def pods_loop(self, path, stop):
        with open(path, "w") as f:
            while not stop.is_set():
                pods = [{"name": f"gpu{r.gpu}", "ready": r.state == "ready",
                         "deleting": r.state == "draining", "phase": r.state}
                        for r in self.reps if r.state != "off"]
                f.write(json.dumps({"t": time.time(), "pods": pods}) + "\n")
                f.flush()
                await asyncio.sleep(2)

    # ---- router
    async def route(self, request):
        ready = self.ready()
        if not ready:
            return web.Response(status=503, text="no ready replica")
        rep = random.choice(ready)
        body = await request.read()
        resp = web.StreamResponse()
        async with self.session.post(f"{rep.url}{request.path}", data=body,
                                     headers={"Content-Type": "application/json"},
                                     timeout=aiohttp.ClientTimeout(total=1800)) as up:
            resp.set_status(up.status)
            resp.content_type = up.content_type
            await resp.prepare(request)
            try:
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
            except (ConnectionResetError,
                    getattr(aiohttp, "ClientConnectionResetError", ConnectionResetError)):
                pass  # client went away (timeout or end of trial); its record shows it
        return resp


async def run_trial(args, policy, load, trial):
    d = os.path.join(args.out, load, policy, f"trial{trial}")
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"skip (done): {d}")
        return
    os.makedirs(d, exist_ok=True)
    seed = 1000 + trial
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"policy": policy, "load": load, "trial": trial, "seed": seed,
                   "schedule": LOADS[load], "max_replicas": MAX_REPLICAS,
                   "scrape": SCRAPE, "sync": SYNC, "stabilization": STABILIZATION,
                   "prestop": PRESTOP, "target": POLICIES[policy] and POLICIES[policy][1],
                   "signal": POLICIES[policy] and POLICIES[policy][0],
                   "testbed": "kaggle T4 x2, real vLLM"}, f, indent=2)
    cl = Cluster(policy, d)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        cl.session = session
        n_static = int(policy.split("-")[1]) if policy.startswith("static-") else 1
        print(f"== {load}/{policy}/trial{trial}: starting {n_static} replica(s)", flush=True)
        await asyncio.gather(*(cl.scale_up(r) for r in cl.reps[:n_static]))
        cl.spec = n_static
        app = web.Application()
        app.router.add_post("/v1/completions", cl.route)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", ROUTER_PORT).start()

        stop = asyncio.Event()
        loops = [asyncio.create_task(cl.pods_loop(os.path.join(d, "pods.jsonl"), stop))]
        if not policy.startswith("static-"):
            loops += [asyncio.create_task(cl.scrape_loop(r, stop)) for r in cl.reps]
            loops.append(asyncio.create_task(cl.hpa_loop(stop)))
        await asyncio.sleep(20)  # let the control loop settle on the idle server
        t_start = time.time()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, os.path.join(ROOT, "cluster", "loadgen.py"),
            "--schedule", *LOADS[load], "--seed", str(seed), "--gateway-port", "0",
            "--base", f"http://127.0.0.1:{ROUTER_PORT}",
            "--out", os.path.join(d, "requests.jsonl"))
        await proc.wait()
        await asyncio.sleep(TAIL)
        t_end = time.time()
        stop.set()
        for t in loops:
            t.cancel()
        await runner.cleanup()
        for r in cl.reps:
            r.stop()
        cl.log(event="trial_end")
        cl.events.close()
    with open(os.path.join(d, "done.json"), "w") as f:
        json.dump({"t_start": t_start, "t_end": t_end, "loadgen_rc": proc.returncode}, f)
    print(f"== done {d} ({t_end - t_start:.0f}s)", flush=True)
    await asyncio.sleep(10)  # let GPU memory free


async def warmup(out):
    """First boot on each GPU: downloads the model and fills the compile cache,
    so that every later start is a warm restart, as in the paper's 73 s median."""
    os.makedirs(out, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        for gpu in (0, 1):
            cl = Cluster.__new__(Cluster)
            cl.session, cl.events = session, open(os.path.join(out, "warmup_events.jsonl"), "a")
            rep = Replica(gpu)
            rep.start(os.path.join(out, f"warmup_gpu{gpu}.log"))
            cl.log(event="create", gpu=gpu)
            await Cluster.wait_health(cl, rep)
            rep.stop()
            cl.events.close()
            print(f"warm-up GPU {gpu} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/val_t4x2")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--policy", choices=sorted(POLICIES))
    ap.add_argument("--load", default="step12", choices=sorted(LOADS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    if args.warmup:
        asyncio.run(warmup(args.out))
        return
    if args.campaign:
        for trial in range(args.trials):
            order = sorted(POLICIES)
            random.Random(trial).shuffle(order)
            for policy in order:
                asyncio.run(run_trial(args, policy, args.load, trial))
        return
    asyncio.run(run_trial(args, args.policy, args.load, args.trial))


if __name__ == "__main__":
    main()
