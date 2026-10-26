#!/usr/bin/env python3
"""
run_trial.py - one autoscaling trial on the kind cluster.

  1. reset: delete ScaledObjects, redeploy the emulator (fresh pod, given
     startup delay), wait until exactly one Ready pod and nothing Terminating
  2. apply the policy (a KEDA ScaledObject) or a static replica count
  3. run loadgen with the given schedule and seed
  4. save: requests, pod lifecycle (polled every 2 s), Prometheus history

Output directory: <out>/<load>/<policy>/trial<k>/

Usage:
  python experiments/run_trial.py --policy queue --load step --trial 0 --out runs/main
  python experiments/run_trial.py --policy static-3 --load ramp --trial 2 --out runs/main
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROM = "http://127.0.0.1:9090"
NS = "llm"

LOADS = {
    # 1 min warm, 2 min burst at ~2.2x single-replica capacity, 3 min tail.
    "step": ["2:60", "20:120", "2:180"],
    # Diurnal-like ramp up to 20 rps and back, 40 s per level.
    "ramp": [f"{r}:40" for r in (2, 5, 9, 14, 18, 20, 18, 14, 9, 5, 2)],
    # Steady load needing ~2 replicas; exposes oscillation.
    "steady": ["12:360"],
}

PROM_QUERIES = {
    "spec_replicas": 'kube_deployment_spec_replicas{namespace="llm",deployment="vllm-sim"}',
    "ready_replicas": 'kube_deployment_status_replicas_ready{namespace="llm",deployment="vllm-sim"}',
    "hpa_desired": 'kube_horizontalpodautoscaler_status_desired_replicas{namespace="llm"}',
    "waiting": 'vllm:num_requests_waiting{namespace="llm"}',
    "running": 'vllm:num_requests_running{namespace="llm"}',
    "kv_cache": 'vllm:kv_cache_usage_perc{namespace="llm"}',
    "success_total": 'vllm:request_success_total{namespace="llm"}',
    "ttft_p90_1m": 'histogram_quantile(0.9, sum(rate(vllm:time_to_first_token_seconds_bucket{namespace="llm"}[1m])) by (le))',
}


def sh(*cmd, check=True, capture=True):
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return r.stdout


def pods():
    data = json.loads(sh("kubectl", "get", "pods", "-n", NS, "-l", "app=vllm-sim", "-o", "json"))
    out = []
    for p in data["items"]:
        ready = any(c["type"] == "Ready" and c["status"] == "True"
                    for c in p["status"].get("conditions", []))
        out.append({"name": p["metadata"]["name"],
                    "created": p["metadata"]["creationTimestamp"],
                    "deleting": p["metadata"].get("deletionTimestamp") is not None,
                    "phase": p["status"].get("phase"), "ready": ready})
    return out


def wait_clean(timeout=900):
    """Wait for exactly one Ready, non-terminating pod and an idle server."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        ps = pods()
        if len(ps) == 1 and ps[0]["ready"] and not ps[0]["deleting"]:
            return
        time.sleep(3)
    raise RuntimeError(f"cluster not clean after {timeout}s: {pods()}")


def reset(startup):
    sh("kubectl", "delete", "scaledobject", "--all", "-n", NS, "--ignore-not-found")
    with open(os.path.join(ROOT, "cluster", "vllm-sim.yaml")) as f:
        manifest = f.read().replace("STARTUP_DURATION", startup)
    subprocess.run(["kubectl", "apply", "-f", "-"], input=manifest, text=True,
                   check=True, capture_output=True)
    sh("kubectl", "scale", "deployment/vllm-sim", "-n", NS, "--replicas=1")
    # Fresh pod per trial so no state carries over between trials. Pods are idle
    # here, so skip the 120 s preStop drain with a near-zero grace period.
    sh("kubectl", "delete", "pods", "-n", NS, "-l", "app=vllm-sim",
       "--grace-period=1", "--wait=false")
    time.sleep(5)
    sh("kubectl", "rollout", "status", "deployment/vllm-sim", "-n", NS, "--timeout=15m")
    wait_clean()


def apply_policy(policy, max_replicas, stabilization, queue_t):
    if policy.startswith("static-"):
        n = int(policy.split("-")[1])
        sh("kubectl", "scale", "deployment/vllm-sim", "-n", NS, f"--replicas={n}")
        sh("kubectl", "rollout", "status", "deployment/vllm-sim", "-n", NS, "--timeout=15m")
        return
    with open(os.path.join(ROOT, "cluster", "policies", f"{policy}.yaml")) as f:
        so = (f.read().replace("MAX_REPLICAS", str(max_replicas))
              .replace("STABILIZATION", str(stabilization))
              .replace("QUEUE_T", str(queue_t)))
    subprocess.run(["kubectl", "apply", "-f", "-"], input=so, text=True,
                   check=True, capture_output=True)
    # Wait until the HPA reads its metric(s), so the trial starts with a live loop.
    t0 = time.time()
    while time.time() - t0 < 180:
        hpa = json.loads(sh("kubectl", "get", "hpa", "-n", NS, "-o", "json"))["items"]
        if hpa and hpa[0].get("status", {}).get("currentMetrics"):
            return
        time.sleep(3)
    raise RuntimeError("HPA never reported metrics")


def poll_pods(stop_evt, path):
    with open(path, "w") as f:
        while not stop_evt.is_set():
            try:
                f.write(json.dumps({"t": time.time(), "pods": pods()}) + "\n")
                f.flush()
            except Exception as e:
                f.write(json.dumps({"t": time.time(), "error": repr(e)}) + "\n")
            stop_evt.wait(2)


def prom_range(query, start, end, step=5):
    url = f"{PROM}/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": query, "start": start, "end": end, "step": step})
    return json.load(urllib.request.urlopen(url, timeout=60))["data"]["result"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--load", required=True, choices=sorted(LOADS))
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--startup", default="73s")
    ap.add_argument("--max-replicas", type=int, default=8)
    ap.add_argument("--stabilization", type=int, default=60)
    ap.add_argument("--queue-threshold", type=int, default=5)
    args = ap.parse_args()

    d = os.path.join(args.out, args.load, args.policy, f"trial{args.trial}")
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"skip (done): {d}")
        return
    os.makedirs(d, exist_ok=True)
    seed = 1000 + args.trial  # same arrivals for every policy within a trial
    config = vars(args) | {"seed": seed, "schedule": LOADS[args.load]}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"== {args.load}/{args.policy}/trial{args.trial}: reset", flush=True)
    reset(args.startup)
    apply_policy(args.policy, args.max_replicas, args.stabilization, args.queue_threshold)
    time.sleep(20)  # let the control loop settle on the idle server

    stop_evt = threading.Event()
    poller = threading.Thread(target=poll_pods,
                              args=(stop_evt, os.path.join(d, "pods.jsonl")))
    poller.start()
    t_start = time.time()
    print(f"== running load {LOADS[args.load]}", flush=True)
    subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"),
                    os.path.join(ROOT, "cluster", "loadgen.py"),
                    "--schedule", *LOADS[args.load], "--seed", str(seed),
                    "--out", os.path.join(d, "requests.jsonl")], check=True)
    # Short tail after the last request completes. Cost (pod-seconds) is
    # compared over the same schedule window for every policy, so this tail
    # only needs to cover Prometheus scrape lag.
    time.sleep(30)
    t_end = time.time()
    stop_evt.set()
    poller.join()

    prom = {}
    for name, q in PROM_QUERIES.items():
        try:
            prom[name] = prom_range(q, t_start - 60, t_end)
        except Exception as e:
            prom[name] = {"error": repr(e)}
    with open(os.path.join(d, "prometheus.json"), "w") as f:
        json.dump(prom, f)
    sh("kubectl", "delete", "scaledobject", "--all", "-n", NS, "--ignore-not-found")
    with open(os.path.join(d, "done.json"), "w") as f:
        json.dump({"t_start": t_start, "t_end": t_end}, f)
    print(f"== done {d} ({t_end - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
