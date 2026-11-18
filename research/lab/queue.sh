#!/bin/sh
# Run every round's arms file in order, one arm at a time (round1.sh skips finished labels).
cd "$(dirname "$0")"
for f in arms/round1.txt arms/round1b.txt arms/round2.txt arms/round2b.txt; do sh round1.sh "$f"; done
echo "queue done $(date +%T)"
