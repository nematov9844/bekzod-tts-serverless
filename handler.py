"""
handler.py — RunPod Serverless Worker for Bekzod Voice 140k F5-TTS
Zero-Defect Speech Bounds Trimming + Crystal Black Background Gate + Studio Baritone Warmth DSP
"""

import os
import io
import re
import gc
import base64
import tempfile
import subprocess
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torchaudio
import soundfile as sf
import numpy as np
import scipy.signal as signal
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import runpod

from f5_tts.model import CFM, DiT
from f5_tts.infer.utils_infer import load_vocoder, target_sample_rate, hop_length

# Local modules
from text_normalizer import normalize_uzbek_text
from phonetic_engine import calculate_phonetic_duration
from audio_stitcher import clean_speech_bounds

# ─────────────────────────────────────────────────────────────────────────────
# 1. INITIALIZATION & ASSET LOADING (RUNS ONCE ON WORKER COLD START)
# ─────────────────────────────────────────────────────────────────────────────

print("[*] Initializing Bekzod TTS Engine on cold start...")

REPO_ID = os.environ.get("HF_REPO_ID", "lynx9844/f5tts-bekzod-200k-uzbek")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")

def get_file(filename: str) -> str:
    local_p = os.path.join(MODEL_DIR, filename)
    if os.path.exists(local_p):
        return local_p
    print(f"[*] Downloading {filename} from Hugging Face ({REPO_ID})...")
    return hf_hub_download(repo_id=REPO_ID, filename=filename, token=HF_TOKEN)

# 1. Vocab
vocab_path = get_file("vocab.txt")
with open(vocab_path, "r", encoding="utf-8") as f:
    raw_vocab = [line.strip("\n") for line in f]
valid_vocab = [l for i, l in enumerate(raw_vocab) if l != "" or i == 0]
vocab_char_map = {l: i for i, l in enumerate(valid_vocab)}
vocab_size = len(vocab_char_map)

# 2. Checkpoint (model_140000.safetensors)
ckpt_path = get_file("model_140000.safetensors")

model = DiT(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_num_embeds=vocab_size,
    text_dim=512,
    mel_dim=100,
    conv_layers=4,
)
cfm = CFM(transformer=model, vocab_char_map=vocab_char_map)

state_dict = load_file(str(ckpt_path))
clean_sd = {}
for k, v in state_dict.items():
    if k in ["initted", "step"]:
        continue
    new_k = k
    if new_k.startswith("ema_model."):
        new_k = new_k[len("ema_model."):]
    if new_k.startswith("transformer."):
        new_k = new_k[len("transformer."):]
    clean_sd[new_k] = v

cfm.transformer.load_state_dict(clean_sd, strict=True)
if DEVICE == "cuda":
    cfm = cfm.half().to(DEVICE).eval()
else:
    cfm = cfm.to(DEVICE).eval()

# 3. Vocoder (Vocos)
vocoder_device = "cpu" if DEVICE == "cuda" else DEVICE
vocoder = load_vocoder("vocos", device=vocoder_device)

# 4. Reference Anchors
def load_anchor(filename: str):
    p = get_file(filename)
    a, sr = torchaudio.load(p)
    if sr != target_sample_rate:
        a = torchaudio.functional.resample(a, sr, target_sample_rate)
    if a.shape[0] > 1:
        a = torch.mean(a, dim=0, keepdim=True)
    if DEVICE == "cuda":
        a = a.half().to(DEVICE)
    mel_len = a.shape[-1] // hop_length
    return a, mel_len

classic_audio, classic_len = load_anchor("ref_classic_baritone.wav")
modern_audio, modern_len = load_anchor("ref_modern_active.wav")

VOICE_PROFILES = {
    "classic": {
        "audio": classic_audio,
        "ref_text": "asosiy qismlari yaponiyada ishlab chiqarilgan elektronikasi janubiy koreyada tayyorlangan.",
        "mel_len": classic_len,
        "speed_factor": 1.0,
    },
    "modern": {
        "audio": modern_audio,
        "ref_text": "transport orqali yevropada yevropa portiga u yerdan temir yo'l.",
        "mel_len": modern_len,
        "speed_factor": 1.02,
    }
}

print(f"[✓] Bekzod TTS Engine initialized successfully on {DEVICE}!")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXACT TEXT PREPROCESSING & ZERO-DEFECT AUDIO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def clean_text_strictly_for_vocab(text: str, vocab_char_map: dict, style: str = "adabiy", already_normalized: bool = False) -> str:
    if not already_normalized:
        norm_text = normalize_uzbek_text(text, style=style).lower()
    else:
        norm_text = text.lower()
    
    # Standalone abbreviations to pronunciation
    norm_text = re.sub(r'\btts\b', 'te te es', norm_text)
    norm_text = re.sub(r'\bai\b', 'ey ay', norm_text)
    norm_text = re.sub(r'\bit\b', 'ay ti', norm_text)
    
    # Phonetic fix for months, loanwords and numbers
    norm_text = re.sub(r'\baudio', 'avdio', norm_text)
    norm_text = re.sub(r'\bsentabr\b|\bsentyabr\b', 'sentiyabr', norm_text)
    norm_text = re.sub(r'\boktyabr\b', 'oktabr', norm_text)
    norm_text = re.sub(r'\bob[- ]havo\b', 'obhavo', norm_text)
    norm_text = re.sub(r'\bsoha', 'sohha', norm_text)

    # Hiatus reinforcement
    norm_text = re.sub(r'\boila', 'oiila', norm_text)
    norm_text = re.sub(r'\bdoira', 'doiira', norm_text)
    norm_text = re.sub(r'\bshoir', 'shoiir', norm_text)
    norm_text = re.sub(r'\brais\b', 'raiis', norm_text)

    # Orphoepic reduction of unstressed 'i' (qalin -> qaln)
    norm_text = re.sub(r'\bqalin([a-z\']*)', r'qaln\1', norm_text)

    # Dashes to spaces
    norm_text = re.sub(r'[-–—_]+', ' ', norm_text)

    # Compound words breakdown
    norm_text = re.sub(r'\bneyrotarmoq', 'neyro tarmoq', norm_text)
    norm_text = re.sub(r'\bnanotexnolog', 'nano texnolog', norm_text)
    norm_text = re.sub(r'\bbiotibbiyot', 'bio tibbiyot', norm_text)
    norm_text = re.sub(r'\bkiberxavfsiz', 'kiber xavfsiz', norm_text)
    norm_text = re.sub(r'\baudiokitob', 'avdio kitob', norm_text)
    norm_text = re.sub(r'\bvideodars', 'video dars', norm_text)
    norm_text = re.sub(r'\bvebsayt', 'veb sayt', norm_text)

    # Number suffixes attachment
    norm_text = re.sub(r'\b(bir|ikki|uch|to\'rt|besh|olti|yetti|sakkiz|to\'qqiz|o\'n|yigirma|o\'ttiz|qirq|ellik|oltmish|yetmish|sakson|sakkson|to\'qson|yuz|ming|million|milliyon|milliard)\s+(dan|ga|da|ni|ning)\b', r'\1\2', norm_text)

    # Normalize apostrophes
    norm_text = unicodedata.normalize('NFC', norm_text)
    norm_text = re.sub(r"[`'ʻʼʽ՚’‘]", "'", norm_text)
    
    # Replace non-vocab punctuation: commas, colons, semicolons become '.' for in-sentence breath pauses
    norm_text = re.sub(r'[,;:]', '.', norm_text)
    norm_text = re.sub(r'[!?]', '.', norm_text)
    norm_text = re.sub(r'\.+', '.', norm_text)
    norm_text = norm_text.replace('"', '').replace('(', '').replace(')', '')
    norm_text = norm_text.replace("w", "v")
    
    # Filter for vocab
    valid_chars = [c for c in norm_text if c in vocab_char_map]
    cleaned = "".join(valid_chars)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def split_sentences_natural(text: str, max_chars: int = 220) -> List[str]:
    """
    Splits text strictly by sentence boundaries (.!? or newlines), preserving full natural cadence.
    Only splits by commas/clauses if a single sentence exceeds max_chars.
    """
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text.strip()]

    chunks = []
    for s in raw_sentences:
        if len(s) <= max_chars:
            chunks.append(s)
        else:
            parts = [p.strip() for p in re.split(r'(?<=[,;:])\s+', s) if p.strip()]
            current = ""
            for p in parts:
                if len(current) + len(p) + 1 <= max_chars:
                    current = f"{current} {p}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = p
            if current:
                chunks.append(current)

    return chunks

def apply_studio_master_dsp(wave: np.ndarray, sr: int = 24000) -> np.ndarray:
    """
    Studio DSP Master (100% Exact to generate_perfect_140k_local.py benchmark):
    1. 150 Hz Low-Shelf (+1.8 dB baritone warmth and body)
    2. 70 Hz Butterworth HPF (clean cut of sub-bass rumble)
    3. 9500 Hz Butterworth LP cut (removes vocoder digital hiss beyond vocal range)
    4. True Peak Normalization (-1 dB / 0.89)
    """
    # 1. Low shelf warmth (+1.8 dB)
    gain = 10 ** (1.8 / 20.0)
    sos_low = signal.butter(2, 150, 'lp', fs=sr, output='sos')
    low_band = signal.sosfiltfilt(sos_low, wave)
    out = wave + (gain - 1.0) * low_band

    # 2. Sub-bass HPF cut (70 Hz)
    sos_hp = signal.butter(2, 70, 'hp', fs=sr, output='sos')
    out = signal.sosfiltfilt(sos_hp, out)

    # 3. High-cut LP filter (9500 Hz) — cleans high-frequency vocoder hiss
    sos_lp = signal.butter(2, 9500, 'lp', fs=sr, output='sos')
    out = signal.sosfiltfilt(sos_lp, out)

    # 4. Peak Limiter (-1 dB / 0.89)
    peak = np.max(np.abs(out))
    if peak > 1e-5:
        out = out * (0.89 / peak)

    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. RUNPOD SERVERLESS HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    job_input = job.get("input", job)
    raw_text = job_input.get("text", "").strip()
    if not raw_text:
        return {"error": "Matn kiritilmagan ('text' bo'sh)", "status": "FAILED"}

    voice = job_input.get("voice", "classic").lower()
    style = job_input.get("style", "adabiy").lower()
    speed = float(job_input.get("speed", 1.0))
    steps = int(job_input.get("steps", 32))
    fmt = job_input.get("format", "mp3").lower()
    seed = job_input.get("seed", None)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    profile = VOICE_PROFILES.get(voice, VOICE_PROFILES["classic"])
    ref_audio = profile["audio"]
    ref_mel_len = profile["mel_len"]
    effective_speed = speed * profile.get("speed_factor", 1.0)
    
    r_text = clean_text_strictly_for_vocab(profile["ref_text"], vocab_char_map)

    # 1. Full Uzbek text normalization (numbers, dates, abbreviations, orthoepy, dialect, tutuq)
    norm_text = normalize_uzbek_text(raw_text, style=style).lower()
    norm_text = re.sub(r'\baudio\b', 'avdio', norm_text)
    norm_text = re.sub(r'\btts\b', 'te te es', norm_text)
    norm_text = re.sub(r'\bai\b', 'ey ay', norm_text)

    # 2. Sentence Splitting strictly by sentence boundaries (keeps commas inside clauses)
    raw_sentences = split_sentences_natural(norm_text, max_chars=240)
    if not raw_sentences:
        return {"error": "Matn tozalangandan so'ng bo'sh qoldi", "status": "FAILED"}

    # 3. Clean each sentence strictly for vocab characters while turning commas into internal in-breath pauses '.'
    text_chunks = []
    for s in raw_sentences:
        c = clean_text_strictly_for_vocab(s, vocab_char_map)
        c = c.strip().strip('.').strip()
        if c:
            text_chunks.append(c)

    if not text_chunks:
        return {"error": "Matn tozalangandan so'ng bo'sh qoldi", "status": "FAILED"}

    # 4. Generate each sentence with 32 steps (clean benchmark quality)
    generated_waves = []
    fade = int(0.015 * target_sample_rate)
    for idx, clean_chunk in enumerate(text_chunks):
        chunk_for_model = clean_chunk + "."
        dur_sec = calculate_phonetic_duration(chunk_for_model, speed_factor=effective_speed)
        target_frames = int(dur_sec * target_sample_rate / hop_length)
        duration = ref_mel_len + target_frames
        full_text = [r_text + " " + chunk_for_model]

        with torch.inference_mode():
            gen, _ = cfm.sample(
                cond=ref_audio,
                text=full_text,
                duration=duration,
                steps=steps,
                cfg_strength=1.55,
                sway_sampling_coef=-1.0,
                seed=seed if seed is not None else (200 + idx)
            )
            gen = gen.to(torch.float32)[:, ref_mel_len:, :].permute(0, 2, 1)
            mel_spec = gen.to(vocoder_device)
            wave_chunk = vocoder.decode(mel_spec).squeeze().cpu().numpy()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            # Clean vocoder onset latency and trailing vocoder air
            chunk_clean = clean_speech_bounds(
                wave_chunk.astype(np.float32),
                sr=target_sample_rate,
                pad_lead_ms=45,
                pad_tail_ms=160
            )
            generated_waves.append(chunk_clean)

    # 5. Natural 240ms human breath pause between sentences
    pause_samples = int(0.24 * target_sample_rate)
    final_pieces = []
    for idx, c in enumerate(generated_waves):
        final_pieces.append(c)
        if idx < len(generated_waves) - 1:
            final_pieces.append(np.zeros(pause_samples, dtype=np.float32))

    full_audio = np.concatenate(final_pieces)

    # 6. Apply Studio Master DSP (150Hz Baritone warmth + 9500Hz cut + peak norm)
    full_audio = apply_studio_master_dsp(full_audio, sr=target_sample_rate)
    total_duration = round(len(full_audio) / target_sample_rate, 2)

    # 7. Encode to MP3 or WAV
    if fmt == "mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, full_audio, target_sample_rate)
            wav_path = tmp_wav.name
        mp3_path = wav_path.replace(".wav", ".mp3")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame",
                "-b:a", "256k", mp3_path
            ], capture_output=True, check=True)
            with open(mp3_path, "rb") as f:
                audio_bytes = f.read()
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            if os.path.exists(mp3_path):
                os.unlink(mp3_path)
    else:
        buf = io.BytesIO()
        sf.write(buf, full_audio, target_sample_rate, format="WAV", subtype="PCM_16")
        audio_bytes = buf.getvalue()

    b64_str = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "status": "COMPLETED",
        "audio_base64": b64_str,
        "duration": total_duration,
        "format": fmt,
        "voice": voice,
        "sample_rate": target_sample_rate,
        "steps": steps
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
