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
pip install --no-cache-dir -q runpod vocos scipy huggingface-hub safetensors soundfile f5-tts google-generativeai

# 3. Fetch latest 1-to-1 pipeline modules from GitHub
echo "[*] Fetching latest pipeline modules from GitHub..."
BASE_URL="https://raw.githubusercontent.com/nematov9844/bekzod-tts-serverless/main"
TS=$(date +%s)
curl -sSfL -H 'Cache-Control: no-cache' "${BASE_URL}/restore_uzbek_orthography.py?v=${TS}" -o /restore_uzbek_orthography.py
curl -sSfL -H 'Cache-Control: no-cache' "${BASE_URL}/text_normalizer.py?v=${TS}" -o /text_normalizer.py
curl -sSfL -H 'Cache-Control: no-cache' "${BASE_URL}/phonetic_engine.py?v=${TS}" -o /phonetic_engine.py
curl -sSfL -H 'Cache-Control: no-cache' "${BASE_URL}/audio_stitcher.py?v=${TS}" -o /audio_stitcher.py
curl -sSfL -H 'Cache-Control: no-cache' "${BASE_URL}/cyrillic_to_latin.py?v=${TS}" -o /cyrillic_to_latin.py
curl -sSfL -H 'Cache-Control: no-cache' "${BASE_URL}/handler.py?v=${TS}" -o /handler.py

# 4. Launch handler
echo "[*] Launching 1-to-1 exact replica handler..."
exec python3 -u /handler.py
