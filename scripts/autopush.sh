#!/usr/bin/env bash
# Every INTERVAL seconds, commit new campaign trial results under runs/main and
# push. Stops after the campaign log reports completion (one final push).
# Usage: scripts/autopush.sh [interval_seconds]
set -uo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${1:-7200}"
LOG=runs/main_campaign.log

push_results() {
  git add runs/main
  if ! git diff --cached --quiet; then
    n=$(find runs/main -name done.json | wc -l | tr -d ' ')
    git commit -q -m "Add campaign trial results ($n of 80 trials complete)"
    git push -q origin main && echo "$(date +%T) pushed ($n trials)"
  else
    echo "$(date +%T) nothing new"
  fi
}

while true; do
  sleep "$INTERVAL"
  push_results
  if grep -q "campaign finished" "$LOG"; then
    echo "campaign finished; final push done"
    break
  fi
done
