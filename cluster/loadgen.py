#!/usr/bin/env python3
"""
loadgen.py - open-loop Poisson load against the cluster Service (127.0.0.1:8080),
following a piecewise-constant rate schedule (e.g. a step burst).

The request mix matches profile_vllm.py, which produced the T4 measurements:
lognormal prompt and output lengths (sigma 0.6, means 256 / 128 tokens, clipped),
ignore_eos so output length is controlled, same seed. The emulator counts one
token per word, so a prompt of n random words is n prompt tokens.

Writes one JSON line per request to --out. Also serves a Prometheus counter
gateway_requests_total (requests sent so far) on --gateway-port, which
Prometheus scrapes from the host; it stands in for a gateway's request counter
and drives the ingress policy (0 disables). With --scrape, also polls /metrics
at 1 Hz into <out>.metrics.jsonl (only meaningful with a single replica, since
the Service picks a pod per connection). Replica counts come from Prometheus.

With --arrivals, replays a JSONL file of {"t", "pin", "pout"} (arrival time in
seconds and token lengths, e.g. a thinned production trace from sim/trace.py)
instead of generating Poisson arrivals; prompts are still random words.

Usage:
  python loadgen.py --schedule 2:60 20:120 2:180 --out runs/step.jsonl
  python loadgen.py --arrivals runs/trace30.arrivals.jsonl --out runs/trace.jsonl
  (rate_rps:duration_s segments, run back to back)
"""
import argparse
import asyncio
import json
import re
import resource
import time

import aiohttp
import numpy as np
from aiohttp import web

# 127.0.0.1, not localhost: each name lookup also costs a file descriptor.
BASE = "http://127.0.0.1:8080"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WORDS = ("time person year way day thing man world life hand part child eye "
         "woman place work week case point government company number group "
         "problem fact").split()
METRICS = {
    "waiting": "vllm:num_requests_waiting",
    "running": "vllm:num_requests_running",
    "kv_cache": "vllm:kv_cache_usage_perc",
}


def raise_fd_limit():
    # macOS shells default to 256 open files; above ~250 in-flight requests the
    # client itself fails ("Too many open files") and caps the offered load.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def sample_lengths(rng, args):
    # Same distribution as profile_vllm.sample_lengths.
    pin = int(np.clip(rng.lognormal(np.log(args.in_mean), 0.6), 8, args.max_in))
    pout = int(np.clip(rng.lognormal(np.log(args.out_mean), 0.6), 8, args.max_out))
    return pin, pout


async def one_request(session, prompt, max_tokens, t_sched, out, base=BASE):
    rec = {"t_sched": t_sched, "t_send": time.time(), "max_tokens": max_tokens,
           "prompt_tokens": None, "completion_tokens": None, "ttft": None,
           "tpot": None, "e2e": None, "error": None}
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    chunk_times = []
    try:
        async with session.post(f"{base}/v1/completions", json=body) as resp:
            if resp.status != 200:
                rec["error"] = f"http {resp.status}"
            else:
                async for raw in resp.content:
                    line = raw.decode(errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    now = time.time()
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        rec["prompt_tokens"] = obj["usage"].get("prompt_tokens")
                        rec["completion_tokens"] = obj["usage"].get("completion_tokens")
                    if obj.get("choices") and obj["choices"][0].get("text"):
                        chunk_times.append(now)
                rec["e2e"] = time.time() - rec["t_send"]
                if chunk_times:
                    rec["ttft"] = chunk_times[0] - rec["t_send"]
                    n = rec["completion_tokens"] or len(chunk_times)
                    if n > 1:
                        rec["tpot"] = (chunk_times[-1] - chunk_times[0]) / (n - 1)
                else:
                    rec["error"] = "no tokens"
    except Exception as e:  # record and keep going; failures are data
        rec["error"] = repr(e)
    out.write(json.dumps(rec) + "\n")


async def scrape_loop(session, stop_evt, path):
    with open(path, "w") as f:
        while not stop_evt.is_set():
            t = time.time()
            row = {"t": t}
            try:
                async with session.get(f"{BASE}/metrics") as r:
                    text = await r.text()
                for key, name in METRICS.items():
                    m = re.search(rf"^{re.escape(name)}{{[^}}]*}} (\S+)$", text, re.M)
                    row[key] = float(m.group(1)) if m else None
            except Exception as e:
                row["error"] = repr(e)
            f.write(json.dumps(row) + "\n")
            await asyncio.sleep(max(0.0, 1.0 - (time.time() - t)))


async def start_gateway_counter(port, counter):
    async def metrics(_request):
        return web.Response(text="# TYPE gateway_requests_total counter\n"
                                 f"gateway_requests_total {counter[0]}\n")
    app = web.Application()
    app.router.add_get("/metrics", metrics)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    return runner


async def main_async(args):
    print(f"open file limit: {raise_fd_limit()}", flush=True)
    sent = [0]
    gateway = await start_gateway_counter(args.gateway_port, sent) if args.gateway_port else None
    segments = [tuple(float(x) for x in s.split(":")) for s in args.schedule or []]
    rng = np.random.default_rng(args.seed)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    # force_close: one connection per request, so kube-proxy picks a pod per
    # request. With keep-alive, pooled connections stay pinned to the pods that
    # existed when they opened and new replicas get little traffic.
    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    tasks = []
    meta = {"schedule": args.schedule, "seed": args.seed, "in_mean": args.in_mean,
            "out_mean": args.out_mean, "max_in": args.max_in,
            "max_out": args.max_out, "t_start": None}
    with open(args.out, "w") as out:
        async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
            stop_evt = asyncio.Event()
            scraper = None
            if args.scrape:
                scraper = asyncio.create_task(scrape_loop(
                    session, stop_evt, args.out.replace(".jsonl", "") + ".metrics.jsonl"))
            t0 = time.time()
            meta["t_start"] = t0
            t_next = t0
            if args.arrivals:
                meta["arrivals"] = args.arrivals
                for a in map(json.loads, open(args.arrivals)):
                    await asyncio.sleep(max(0.0, t0 + a["t"] - time.time()))
                    prompt = " ".join(rng.choice(WORDS, size=a["pin"]))
                    sent[0] += 1
                    tasks.append(asyncio.create_task(
                        one_request(session, prompt, a["pout"], a["t"], out, args.base)))
                segments = []
            for rate, dur in segments:
                seg_end = t_next + dur
                print(f"[{t_next - t0:6.0f}s] rate {rate} rps for {dur:.0f}s", flush=True)
                while True:
                    t_next += rng.exponential(1.0 / rate)
                    if t_next >= seg_end:
                        t_next = seg_end
                        break
                    await asyncio.sleep(max(0.0, t_next - time.time()))
                    pin, pout = sample_lengths(rng, args)
                    prompt = " ".join(rng.choice(WORDS, size=pin))
                    sent[0] += 1
                    tasks.append(asyncio.create_task(
                        one_request(session, prompt, pout, t_next - t0, out, args.base)))
            print(f"schedule done, draining {sum(not t.done() for t in tasks)} requests",
                  flush=True)
            await asyncio.gather(*tasks)
            meta["t_end"] = time.time()
            if scraper:
                stop_evt.set()
                await scraper
    if gateway:
        await gateway.cleanup()
    with open(args.out.replace(".jsonl", "") + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {len(tasks)} requests to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrivals", help="JSONL of {t, pin, pout} to replay instead of --schedule")
    ap.add_argument("--schedule", nargs="+",
                    help="rate_rps:duration_s segments, e.g. 2:60 20:120 2:180")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scrape", action="store_true",
                    help="poll /metrics at 1 Hz (single replica only)")
    ap.add_argument("--in-mean", type=int, default=256)
    ap.add_argument("--out-mean", type=int, default=128)
    ap.add_argument("--max-in", type=int, default=1024)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gateway-port", type=int, default=9400)
    ap.add_argument("--base", default=BASE, help="server URL (default: the kind Service)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
