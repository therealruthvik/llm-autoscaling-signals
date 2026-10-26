#!/usr/bin/env python3
"""
profile_vllm.py - measure real vLLM behavior on a single GPU (e.g. Kaggle/Colab T4).

Produces the measured parameters that feed the autoscaling simulator/emulator:
  1. Cold start breakdown (spawn -> /health ready), repeated N times
  2. Load sweep: Poisson arrivals at several req/s, per-request TTFT / TPOT / E2E
  3. Server-side metrics trace (queue depth, running reqs, KV-cache usage) at 1 Hz

Outputs (in --out dir):
  coldstart.json        list of cold start timings
  requests_rate<R>.jsonl one line per request
  metrics_rate<R>.jsonl  1 Hz scrape of /metrics
  summary.csv            one row per rate level
  env.json               GPU, driver, vllm version (for the paper's setup section)

Usage (Kaggle notebook cell):
  !pip install -q vllm aiohttp numpy
  !python profile_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct --out results \
        --rates 1 2 4 8 16 --duration 60 --coldstart-trials 3

Notes:
  - T4 has no bf16 -> we pass --dtype half.
  - Output length is forced with ignore_eos so token counts are controlled.
  - Image pull time is NOT measured here (no container runtime on Kaggle);
    measure that separately on your kind/EKS cluster and add to the model.
  - Trial 1 of cold start has a cold disk/page cache; later trials are warmer.
    Report both. Do not average them blindly.
"""
import argparse
import asyncio
import csv
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time

import aiohttp
import numpy as np

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"

# vLLM renamed some metrics across versions; try all known names.
METRIC_ALIASES = {
    "waiting": ["vllm:num_requests_waiting"],
    "running": ["vllm:num_requests_running"],
    "kv_cache": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
}


# ---------------------------------------------------------------- server mgmt
def start_server(model, max_len, gpu_util, log_path):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(PORT),
        "--dtype", "half",
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(gpu_util),
    ]
    logf = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    return proc, t0, logf


async def wait_ready(proc, t0, timeout=900):
    """Poll /health. Returns seconds from spawn to ready."""
    async with aiohttp.ClientSession() as s:
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                raise RuntimeError("vLLM exited early; check server log")
            try:
                async with s.get(f"{BASE}/health",
                                 timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        return time.time() - t0
            except Exception:
                pass
            await asyncio.sleep(0.25)
    raise TimeoutError("vLLM did not become ready")


def stop_server(proc, logf):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    logf.close()
    time.sleep(5)  # let GPU memory free


def parse_log_phases(log_path):
    """Best-effort extraction of phase timings from vLLM log lines.
    Log formats change by version, so this returns whatever it can find."""
    phases = {}
    try:
        text = open(log_path, errors="ignore").read()
    except Exception:
        return phases
    pats = {
        "weights_load_s": r"Loading (?:model )?weights took ([\d.]+) seconds",
        "model_load_s": r"Model loading took [\d.]+ GiB.*?([\d.]+) seconds",
        "init_engine_s": r"init engine.*?took ([\d.]+) seconds",
        "graph_capture_s": r"Graph capturing finished in ([\d.]+) secs",
        "profile_run_s": r"profile.*?took ([\d.]+) seconds",
        "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "gpu_blocks": r"# GPU blocks: (\d+)",
        "max_concurrency_x": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    }
    for k, p in pats.items():
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if m:
            phases[k] = float(m.group(1).replace(",", ""))
    return phases


# ------------------------------------------------------------------- metrics
def parse_prom(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+eE0-9.naNinf]+)$", line)
        if not m:
            continue
        name, val = m.group(1), m.group(3)
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


async def scrape_loop(session, stop_evt, path):
    with open(path, "w") as f:
        while not stop_evt.is_set():
            t = time.time()
            try:
                async with session.get(f"{BASE}/metrics",
                                       timeout=aiohttp.ClientTimeout(total=2)) as r:
                    prom = parse_prom(await r.text())
                row = {"t": t}
                for key, names in METRIC_ALIASES.items():
                    for n in names:
                        if n in prom:
                            row[key] = prom[n]
                            break
                f.write(json.dumps(row) + "\n")
            except Exception:
                pass
            await asyncio.sleep(max(0, 1.0 - (time.time() - t)))


# --------------------------------------------------------------- load driver
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()


_TOKENIZER = None


def load_tokenizer(model):
    """Use the model's own tokenizer so prompt lengths are exact. Falls back to
    a word-count approximation if transformers is unavailable."""
    global _TOKENIZER
    try:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model)
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({e}); prompt lengths approximate",
              flush=True)


def make_prompt(rng, n_tokens):
    # Oversample words, then trim to exactly n_tokens with the real tokenizer.
    # Without trimming, these words tokenize to >1 token each and long prompts
    # can overflow --max-model-len, producing failures that are not real load.
    words = " ".join(rng.choice(WORDS, size=max(4, n_tokens)))
    if _TOKENIZER is None:
        return " ".join(words.split()[:max(4, int(n_tokens / 2))])
    ids = _TOKENIZER.encode(words, add_special_tokens=False)[:n_tokens]
    return _TOKENIZER.decode(ids)


def sample_lengths(rng, in_mu, out_mu, max_in, max_out):
    pin = int(np.clip(rng.lognormal(np.log(in_mu), 0.6), 8, max_in))
    pout = int(np.clip(rng.lognormal(np.log(out_mu), 0.6), 8, max_out))
    return pin, pout


async def one_request(session, model, prompt, max_tokens, t_sched, results):
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    rec = {"t_sched": t_sched, "max_tokens": max_tokens, "ok": False}
    t_send = time.time()
    rec["t_send"] = t_send
    chunk_times = []
    try:
        async with session.post(f"{BASE}/v1/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=300)) as r:
            if r.status != 200:
                rec["status"] = r.status
                results.append(rec)
                return
            async for raw in r.content:
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
        t_end = time.time()
        if chunk_times:
            rec["ok"] = True
            rec["ttft"] = chunk_times[0] - t_send
            rec["e2e"] = t_end - t_send
            n = rec.get("completion_tokens") or len(chunk_times)
            rec["tpot"] = ((chunk_times[-1] - chunk_times[0]) / max(1, n - 1)
                           if n > 1 else None)
    except Exception as e:
        rec["error"] = repr(e)
    results.append(rec)


async def run_rate(model, rate, duration, seed, in_mu, out_mu, max_in, max_out,
                   out_dir, tag=None):
    tag = tag or f"rate{rate:g}"
    rng = np.random.default_rng(seed)
    results = []
    stop_evt = asyncio.Event()
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        scraper = asyncio.create_task(
            scrape_loop(session, stop_evt,
                        os.path.join(out_dir, f"metrics_{tag}.jsonl")))
        tasks = []
        t0 = time.time()
        t_next = rng.exponential(1.0 / rate)  # Poisson arrivals
        while t_next < duration:
            delay = t0 + t_next - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            pin, pout = sample_lengths(rng, in_mu, out_mu, max_in, max_out)
            tasks.append(asyncio.create_task(
                one_request(session, model, make_prompt(rng, pin), pout,
                            t_next, results)))
            t_next += rng.exponential(1.0 / rate)
        await asyncio.gather(*tasks)  # drain
        wall = time.time() - t0
        stop_evt.set()
        await scraper
    with open(os.path.join(out_dir, f"requests_{tag}.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results, wall


def summarize(rate, results, duration, wall):
    ok = [r for r in results if r.get("ok")]
    def pct(key, q):
        v = [r[key] for r in ok if r.get(key) is not None]
        return float(np.percentile(v, q)) if v else None
    toks = sum((r.get("completion_tokens") or 0) for r in ok)
    return {
        "rate_rps": rate,
        "sent": len(results),
        "ok": len(ok),
        "fail": len(results) - len(ok),
        "ttft_p50": pct("ttft", 50), "ttft_p95": pct("ttft", 95), "ttft_p99": pct("ttft", 99),
        "tpot_p50": pct("tpot", 50), "tpot_p95": pct("tpot", 95), "tpot_p99": pct("tpot", 99),
        "e2e_p50": pct("e2e", 50), "e2e_p99": pct("e2e", 99),
        # Divide by real wall time (arrival window + drain). Dividing by the
        # nominal duration overstates throughput once the server is overloaded.
        "wall_s": wall,
        "throughput_tok_s": toks / wall,
        "goodput_req_s": len(ok) / wall,
    }


# --------------------------------------------------------------------- main
def capture_env(model, args):
    env = {"python": sys.version, "platform": platform.platform(),
           "model": model, "args": vars(args)}
    try:
        env["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        env["nvidia_smi"] = f"unavailable: {e}"
    try:
        import vllm
        env["vllm_version"] = vllm.__version__
    except Exception:
        env["vllm_version"] = "unknown"
    return env


async def main_async(args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "env.json"), "w") as f:
        json.dump(capture_env(args.model, args), f, indent=2)

    load_tokenizer(args.model)

    # ---- 1. cold start trials
    # Trial 0 includes the HF model download if not cached; later trials load
    # from local disk. Report them separately.
    cold = []
    server = None
    for i in range(args.coldstart_trials):
        log_path = os.path.join(args.out, f"server_coldstart_{i}.log")
        proc, t0, logf = start_server(args.model, args.max_model_len,
                                      args.gpu_util, log_path)
        last = i == args.coldstart_trials - 1
        try:
            ready_s = await wait_ready(proc, t0)
        except BaseException:
            stop_server(proc, logf)  # never leave a server holding the GPU
            raise
        logf.flush()
        rec = {"trial": i, "spawn_to_ready_s": ready_s,
               **parse_log_phases(log_path)}
        cold.append(rec)
        print(f"[coldstart {i}] {rec}", flush=True)
        if last:
            server = (proc, logf)  # keep last one up for load sweep
        else:
            stop_server(proc, logf)
    with open(os.path.join(args.out, "coldstart.json"), "w") as f:
        json.dump(cold, f, indent=2)

    # ---- 2. warmup then rate sweep
    # summary.csv is rewritten after every level so a Kaggle timeout or crash
    # mid-sweep still leaves usable data.
    summary = []
    summary_path = os.path.join(args.out, "summary.csv")
    try:
        await run_rate(args.model, 1, 10, 0, args.in_mean, args.out_mean,
                       args.max_in, args.max_out, args.out, tag="warmup")
        for rate in args.rates:
            print(f"[sweep] rate={rate:g} rps for {args.duration}s", flush=True)
            res, wall = await run_rate(args.model, rate, args.duration, args.seed,
                                       args.in_mean, args.out_mean,
                                       args.max_in, args.max_out, args.out)
            row = summarize(rate, res, args.duration, wall)
            summary.append(row)
            print(row, flush=True)
            with open(summary_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                w.writeheader()
                w.writerows(summary)
            await asyncio.sleep(10)  # let queue drain between levels
    finally:
        if server:
            stop_server(*server)
    print("done ->", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results")
    ap.add_argument("--rates", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--coldstart-trials", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--in-mean", type=int, default=256)
    ap.add_argument("--out-mean", type=int, default=128)
    ap.add_argument("--max-in", type=int, default=1024)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()