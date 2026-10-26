#!/usr/bin/env python3
"""Generate one KEDA ScaledObject per autoscaling policy (rerun after edits).

Placeholders filled in at apply time by experiments/run_trial.py:
MAX_REPLICAS, STABILIZATION, QUEUE_T.
Thresholds marked "80% of measured capacity" are the signal's mean value on the
T4 at 6.4 req/s = 0.8 x saturated single-replica capacity (7.8-8.2 completions/s
at 24 and 32 req/s offered, data/results_run{1,2,3}), linearly interpolated
between the 4 and 8 req/s levels.
"""
HEAD = """# {desc}
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-sim-{name}
  namespace: llm
spec:
  scaleTargetRef:
    name: vllm-sim
  minReplicaCount: 1
  maxReplicaCount: MAX_REPLICAS
  pollingInterval: 5
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: STABILIZATION
  triggers:
"""
PROM = "http://prometheus-server.monitoring.svc.cluster.local:80"
NS = 'namespace="llm"'


def trig(query, threshold, mtype="AverageValue"):
    return f"""    - type: prometheus
      metricType: {mtype}
      metadata:
        serverAddress: {PROM}
        query: {query}
        threshold: "{threshold}"
"""


POLICIES = {
    "queue": ("Queue depth: waiting requests per replica, threshold QUEUE_T (5 = Liu & Hu 2026 KEDA baseline).",
              [trig(f'sum(vllm:num_requests_waiting{{{NS}}})', "QUEUE_T")]),
    "running": ("Batch size: running requests per replica, target 50 = value at 80% of measured T4 capacity (GKE batch-size signal).",
                [trig(f'sum(vllm:num_requests_running{{{NS}}})', "50")]),
    "kv-cal": ("KV-cache usage per replica, target 0.06 = value at 80% of measured T4 capacity.",
               [trig(f'sum(vllm:kv_cache_usage_perc{{{NS}}})', "0.06")]),
    "kv-default": ("KV-cache usage per replica, target 0.8 (common default, llm-d docs).",
                   [trig(f'sum(vllm:kv_cache_usage_perc{{{NS}}})', "0.8")]),
    "ttft": ("TTFT SLO signal: p90 TTFT over 1m, target 1s (HPA Value: desired = ceil(replicas * p90 / 1s)).",
             [trig(f'histogram_quantile(0.9, sum(rate(vllm:time_to_first_token_seconds_bucket{{{NS}}}[1m])) by (le))', "1", "Value")]),
    "qps": ("Throughput: completed requests/s per replica, target 6.4 = 80% of measured T4 capacity (Liu & Hu HPA-QPS baseline uses 8).",
            [trig(f'sum(rate(vllm:request_success_total{{{NS}}}[1m]))', "6.4")]),
    "combined": ("Saturation-style combination (llm-d WVA): queue per replica > QUEUE_T OR KV per replica > 0.8; HPA takes the max.",
                 [trig(f'sum(vllm:num_requests_waiting{{{NS}}})', "QUEUE_T"),
                  trig(f'sum(vllm:kv_cache_usage_perc{{{NS}}})', "0.8")]),
}

if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for name, (desc, trigs) in POLICIES.items():
        with open(os.path.join(here, f"{name}.yaml"), "w") as f:
            f.write(HEAD.format(desc=desc, name=name) + "".join(trigs))
    print("wrote", ", ".join(sorted(POLICIES)))
