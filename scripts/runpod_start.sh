#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/model-video-image}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

export HF_HOME
export HUGGINGFACE_HUB_CACHE
export HF_HUB_DISABLE_XET

cd "$APP_DIR"

echo "[start] HF_HOME=$HF_HOME"
echo "[start] HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "[start] HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "[start] launching uvicorn on ${HOST}:${PORT}"

exec python -m uvicorn main:app --host "$HOST" --port "$PORT"
