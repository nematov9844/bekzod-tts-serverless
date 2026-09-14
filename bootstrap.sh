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

# 2. Python dependencies
echo "[*] Ensuring Python packages..."
pip install --no-cache-dir -q runpod vocos scipy huggingface-hub safetensors soundfile f5-tts

# 3. Fetch latest 1-to-1 pipeline modules from GitHub
echo "[*] Fetching latest pipeline modules from GitHub..."
BASE_URL="https://raw.githubusercontent.com/nematov9844/bekzod-tts-serverless/main"
curl -sSfL "${BASE_URL}/restore_uzbek_orthography.py" -o /restore_uzbek_orthography.py
curl -sSfL "${BASE_URL}/text_normalizer.py" -o /text_normalizer.py
curl -sSfL "${BASE_URL}/phonetic_engine.py" -o /phonetic_engine.py
curl -sSfL "${BASE_URL}/audio_stitcher.py" -o /audio_stitcher.py
curl -sSfL "${BASE_URL}/handler.py" -o /handler.py

# 4. Launch handler
echo "[*] Launching 1-to-1 exact replica handler..."
exec python3 -u /handler.py
