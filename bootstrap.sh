#!/usr/bin/env bash
set -e

echo "[*] Starting Bekzod TTS Serverless Worker Bootstrap..."

# 1. Dependencies
if [ ! -f /tmp/installed.flag ]; then
    echo "[*] Installing ffmpeg..."
    apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg >/dev/null 2>&1 || true
    echo "[*] Installing Python packages..."
    pip install --no-cache-dir -q runpod vocos scipy huggingface-hub safetensors soundfile f5-tts
    touch /tmp/installed.flag
    echo "[✓] Environment ready."
fi

# 2. Fetch latest handler
echo "[*] Fetching handler.py..."
curl -sSfL https://raw.githubusercontent.com/nematov9844/bekzod-tts-serverless/main/handler.py -o /handler.py

# 3. Start handler
echo "[*] Launching handler..."
exec python -u /handler.py
