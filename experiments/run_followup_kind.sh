#!/usr/bin/env bash
# Second kind campaign (2026-10-30): ingress and static-floor policies on the
# main loads, the two-replica validation counterpart (step12, max 2 replicas),
# and the 30-minute Azure trace window. Each campaign is resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY experiments/run_campaign.py --out runs/followup_kind_v1 --trials 5 \
    --policies ingress running-floor2 --loads step ramp
$PY experiments/run_campaign.py --out runs/val_kind_step12_v1 --trials 3 --max-replicas 2 \
    --policies queue running kv-cal static-1 static-2 --loads step12
$PY experiments/run_campaign.py --out runs/trace30_kind_v1 --trials 2 \
    --policies queue running ingress static-3 --loads trace30
echo "all kind campaigns finished"
