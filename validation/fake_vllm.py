#!/usr/bin/env python3
"""fake_vllm.py - tiny stand-in for a vLLM server, used only to self-test
two_replica_t4.py without a GPU (FAKE_VLLM=1). Not a model of vLLM: fixed
startup delay, MAX_SEQS concurrent streams, fixed time per token."""
import argparse
import asyncio
import json
import time

from aiohttp import web

STARTUP = 8.0
MAX_SEQS = 8
TOKEN_S = 0.01

state = {"waiting": 0, "running": 0, "t0": time.time()}
sem = asyncio.Semaphore(MAX_SEQS)


async def health(_):
    return web.Response(status=200 if time.time() - state["t0"] > STARTUP else 503)


async def metrics(_):
    return web.Response(text=(f'vllm:num_requests_waiting{{engine="0"}} {state["waiting"]}\n'
                              f'vllm:num_requests_running{{engine="0"}} {state["running"]}\n'
                              f'vllm:kv_cache_usage_perc{{engine="0"}} {state["running"] / 100}\n'))


async def completions(request):
    body = await request.json()
    n = body["max_tokens"]
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    await resp.prepare(request)
    state["waiting"] += 1
    async with sem:
        state["waiting"] -= 1
        state["running"] += 1
        for _ in range(n):
            await asyncio.sleep(TOKEN_S)
            await resp.write(b'data: {"choices":[{"text":"x"}]}\n\n')
        state["running"] -= 1
    usage = {"usage": {"prompt_tokens": 1, "completion_tokens": n}, "choices": []}
    await resp.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
    await resp.write_eof()
    return resp


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int)
app = web.Application()
app.router.add_get("/health", health)
app.router.add_get("/metrics", metrics)
app.router.add_post("/v1/completions", completions)
web.run_app(app, host="127.0.0.1", port=ap.parse_args().port, print=None)
