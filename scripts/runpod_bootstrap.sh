#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/model-video-image}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

cd "$APP_DIR"

echo "[bootstrap] upgrading pip"
python -m pip install --upgrade pip

echo "[bootstrap] installing torch/cu124 stack"
python -m pip install \
  torch==2.5.1+cu124 \
  torchvision==0.20.1+cu124 \
  torchaudio==2.5.1+cu124 \
  --index-url "$TORCH_INDEX_URL"

echo "[bootstrap] installing project requirements"
python -m pip install -r requirements.txt

echo "[bootstrap] done"
echo "[bootstrap] start server with: ./scripts/runpod_start.sh"
