#!/bin/sh
# Runs the SOLO scaling arms files in order (round1.sh skips finished labels).
cd "$(dirname "$0")"
for f in "$@"; do sh round1.sh "$f"; done
echo "solo queue done $(date +%T)"
