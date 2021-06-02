#!/usr/bin/env bash
set -euo pipefail


cd data/ace_cdaweb/mag

for file in *.tar.gz; do
    tar -zxvf "$file"
done
