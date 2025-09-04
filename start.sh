#!/usr/bin/env bash
set -e

MODELS_DIR="/app/official_models"
MODELS_URL="https://github.com/jkb31052002/CITSwAIVISION_OCR/releases/download/v1-models/official_models.tar.gz"

if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
  echo "[models] not found; downloading..."
  mkdir -p "$MODELS_DIR"
  cd /app
  curl -L "$MODELS_URL" -o models_archive
  # auto-detect archive type
  if file models_archive | grep -qi 'gzip'; then
    tar -xzf models_archive -C /app
  elif file models_archive | grep -qi 'zip'; then
    apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*
    unzip -o models_archive -d /app
  else
    echo "Unknown archive format"; exit 1
  fi
  rm -f models_archive
  echo "[models] ready."
fi

echo "[server] starting gunicorn..."
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-8080} app:app
