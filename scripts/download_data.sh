#!/usr/bin/env bash
# Downloads the PaySim dataset (Kaggle: ealaxi/paysim1, licence CC BY-SA 4.0).
# Before running, log in once with: .venv/bin/kaggle auth login
set -euo pipefail
RAW_DIR="data/raw"
TARGET="$RAW_DIR/paysim.csv"
mkdir -p "$RAW_DIR"
if [[ -f "$TARGET" ]]; then
  echo "paysim.csv already present, skipping"
else
  .venv/bin/kaggle datasets download -d ealaxi/paysim1 -p "$RAW_DIR" --unzip
  mv "$RAW_DIR"/PS_*_log.csv "$TARGET"
fi
if [[ ! -f "$TARGET" ]]; then
  echo "Download failed. Run '.venv/bin/kaggle auth login' and try again." >&2
  exit 1
fi
ls -lh "$RAW_DIR"
