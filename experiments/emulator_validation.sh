#!/usr/bin/env bash
# Emulator validation: drive ONE emulator replica (no autoscaler) with the same
# rate sweep and request mix as the T4 profiling runs, so emulator behavior can
# be compared to measured vLLM behavior rate by rate.
# Usage: experiments/emulator_validation.sh <out_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:?out dir}"
mkdir -p "$OUT"
cp cluster/vllm-sim.yaml "$OUT/vllm-sim.yaml"  # exact emulator config used

kubectl delete scaledobject --all -n llm --ignore-not-found
cluster/03_deploy_sim.sh "${STARTUP:-73s}"

for R in 1 2 4 8 16 24 32; do
  .venv/bin/python cluster/loadgen.py --schedule "${R}:60" --scrape \
      --out "$OUT/requests_rate${R}.jsonl"
done
echo "done: $OUT"
