#!/usr/bin/env bash
set -euo pipefail

PATHS=$( ls ../data/ace/daily/*_ace_*_1m.txt )

[[ $# -gt 0 ]] && PATHS=$@

for file in $PATHS; do
    sed --in-place '/# YR/s/^#//g' "$file" 
    sed --in-place "/#\|:/d" "$file"
done
