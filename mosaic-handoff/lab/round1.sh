#!/bin/sh
# Round 1: every arm 120 s on the 10m preset, seq 128, batch 8, MPS, run one at a time.
# Usage: sh round1.sh [arms-file]   (one "method k=v ... --label X [--init ...]" per line)
# An arm may override the budget or preset, e.g. "... --seconds 300 --preset 100m".
cd "$(dirname "$0")"
PY=${PY:-python}
ARMS=${1:-arms/round1.txt}
mkdir -p logs
grep -v '^#' "$ARMS" | grep -v '^$' | while IFS= read -r arm; do
  label=$(echo "$arm" | sed -n 's/.*--label \([^ ]*\).*/\1/p')
  if [ -f "results/$label.json" ]; then echo "skip $label"; continue; fi
  echo "=== $label $(date +%T)"
  # shellcheck disable=SC2086
  $PY harness.py --seconds 120 $arm > "logs/$label.log" 2>&1 || echo "FAILED $label"
  tail -n 2 "logs/$label.log"
done
echo "round1 done $(date +%T)"
