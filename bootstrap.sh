#!/usr/bin/env bash
set -eo pipefail

echo "=========================================="
echo "[*] BEKZOD TTS SERVERLESS WORKER BOOTSTRAP"
echo "=========================================="
echo "Time: $(date)"

# 1. ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[*] Installing ffmpeg..."
    apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg >/dev/null 2>&1 || true
fi

# 2. python dependencies
echo "[*] Ensuring Python packages..."
pip install --no-cache-dir -q runpod vocos scipy huggingface-hub safetensors soundfile f5-tts

# 3. download latest handler.py
echo "[*] Fetching handler.py from GitHub..."
curl -sSfL https://raw.githubusercontent.com/nematov9844/bekzod-tts-serverless/main/handler.py -o /handler.py

# 4. launch handler
echo "[*] Launching handler.py with Python..."
exec python3 -u /handler.py
