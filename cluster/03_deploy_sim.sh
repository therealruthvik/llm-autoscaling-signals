#!/usr/bin/env bash
# Deploy the emulator at 1 replica. Usage: ./03_deploy_sim.sh [startup_duration]
# startup_duration defaults to 73s (median measured warm cold start); it is
# swept for RQ2.
set -euo pipefail
cd "$(dirname "$0")"
STARTUP="${1:-73s}"

kubectl create namespace llm --dry-run=client -o yaml | kubectl apply -f -
sed "s/STARTUP_DURATION/${STARTUP}/" vllm-sim.yaml | kubectl apply -f -
kubectl scale deployment/vllm-sim -n llm --replicas=1

echo "waiting for rollout (startup-duration ${STARTUP})..."
kubectl rollout status deployment/vllm-sim -n llm --timeout=10m

curl -s http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"hello","max_tokens":16}'
echo
curl -s http://127.0.0.1:8080/metrics | grep -E '^vllm:(num_requests_(waiting|running)|kv_cache_usage_perc)'
