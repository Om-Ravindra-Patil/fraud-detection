#!/usr/bin/env bash
# Downloads the IEEE-CIS training files. Before running:
#   1. Accept the competition rules at https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#   2. Put your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600)
set -euo pipefail
RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"
for f in train_transaction.csv train_identity.csv; do
  if [[ -f "$RAW_DIR/$f" ]]; then
    echo "$f already present, skipping"
    continue
  fi
  .venv/bin/kaggle competitions download -c ieee-fraud-detection -f "$f" -p "$RAW_DIR"
  if [[ -f "$RAW_DIR/$f.zip" ]]; then
    unzip -o -q "$RAW_DIR/$f.zip" -d "$RAW_DIR" && rm "$RAW_DIR/$f.zip"
  fi
done
ls -lh "$RAW_DIR"
