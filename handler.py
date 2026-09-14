"""
handler.py — RunPod Serverless Worker for Bekzod Voice 140k F5-TTS
100% 1-to-1 Server Benchmark Replica (Phonetic Engine + Text Normalizer + Audio Stitcher + Studio DSP)
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
from audio_stitcher import stitch_chunks_zero_defect

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
    mel_len = cfm.mel_spec(a).shape[-1] if hasattr(cfm, 'mel_spec') else a.shape[-1] // hop_length
    return a, mel_len

modern_audio, modern_len = load_anchor("ref_modern_active.wav")
classic_audio, classic_len = load_anchor("ref_classic_baritone.wav")

VOICE_PROFILES = {
    "modern": {
        "audio": modern_audio,
        "ref_text": "transport orqali yevropada yevropa portiga u yerdan temir yo'l.",
        "mel_len": modern_len,
        "speed_factor": 1.05,
    },
    "classic": {
        "audio": classic_audio,
        "ref_text": "asosiy qismlari yaponiyada ishlab chiqarilgan elektronikasi janubiy koreyada tayyorlangan.",
        "mel_len": classic_len,
        "speed_factor": 0.98,
    }
}

print(f"[✓] Bekzod TTS Engine initialized successfully on {DEVICE}!")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXACT 1-TO-1 TEXT PREPROCESSING & DSP
# ─────────────────────────────────────────────────────────────────────────────

def clean_text_strictly_for_vocab(text: str, vocab_char_map: dict, style: str = "adabiy", already_normalized: bool = False) -> str:
    if not already_normalized:
        norm_text = normalize_uzbek_text(text, style=style).lower()
    else:
        norm_text = text.lower()
    
    # Standalone English/tech abbreviations to pronunciation
    norm_text = re.sub(r'\btts\b', 'te te es', norm_text)
    norm_text = re.sub(r'\bai\b', 'ey ay', norm_text)
    norm_text = re.sub(r'\bit\b', 'ay ti', norm_text)
    
    # Phonetic fix for months, loanwords and numbers
    norm_text = re.sub(r'\baudio', 'avdio', norm_text)
    norm_text = re.sub(r'\bsentabr\b', 'sentiyabr', norm_text)
    norm_text = re.sub(r'\bsentyabr\b', 'sentiyabr', norm_text)
    norm_text = re.sub(r'\boktyabr\b', 'oktabr', norm_text)
    norm_text = re.sub(r'\bob[- ]havo\b', 'obhavo', norm_text)
    norm_text = re.sub(r'\bsoha', 'sohha', norm_text)

    # Hiatus reinforcement
    norm_text = re.sub(r'\boila', 'oiila', norm_text)
    norm_text = re.sub(r'\bdoira', 'doiira', norm_text)
    norm_text = re.sub(r'\bshoir', 'shoiir', norm_text)
    norm_text = re.sub(r'\brais\b', 'raiis', norm_text)

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
    
    # Replace non-vocab punctuation
    norm_text = norm_text.replace("!", ".").replace(":", ".").replace(";", ".").replace('"', '').replace('(', '').replace(')', '')
    
    # Filter for vocab
    valid_chars = [c for c in norm_text if c in vocab_char_map]
    cleaned = "".join(valid_chars)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def split_clause_smart(sent: str, max_chars: int = 200) -> List[str]:
    sent = sent.strip()
    if len(sent) <= max_chars:
        return [sent]
    if re.search(r'[,;:]\s+', sent):
        parts = [p.strip() for p in re.split(r'(?<=[,;:])\s+', sent) if p.strip()]
        res = []
        for p in parts:
            res.extend(split_clause_smart(p, max_chars))
        return res

    words = sent.split()
    mid = len(words) // 2
    p1 = ' '.join(words[:mid]).rstrip(',') + ','
    p2 = ' '.join(words[mid:])
    res = []
    res.extend(split_clause_smart(p1, max_chars))
    res.extend(split_clause_smart(p2, max_chars))
    return res

def apply_studio_dsp(wave: np.ndarray, sr: int = 24000, profile: str = "modern") -> np.ndarray:
    """
    Enhanced Studio DSP:
    1. 70 Hz Butterworth HPF (sub-bass rumble and mic floor cut)
    2. Profile EQ (Classic: 150Hz warmth; Modern: 8500Hz smooth high-cut)
    3. Smooth Studio Noise Gate (eliminates background hiss during quiet intervals)
    4. Equal Loudness Normalization (levels Modern & Classic to identical volume)
    5. True Peak Limiter (-1 dB / 0.89)
    """
    out = wave.copy()

    # 1. 70 Hz Rumble HPF
    sos_hp = signal.butter(2, 70, 'hp', fs=sr, output='sos')
    out = signal.sosfiltfilt(sos_hp, out)

    # 2. Profile EQ
    if profile in ["classic", "bekzod"]:
        gain = 10 ** (1.8 / 20.0)
        sos_low = signal.butter(2, 150, 'lp', fs=sr, output='sos')
        low_band = signal.sosfiltfilt(sos_low, out)
        out = out + (gain - 1.0) * low_band
    else:
        sos_lp = signal.butter(2, 8500, 'lp', fs=sr, output='sos')
        out = signal.sosfiltfilt(sos_lp, out)

    # 3. Smooth Studio Noise Gate (10ms attack, 220ms safe release)
    attack_coeff = np.exp(-1.0 / (0.010 * sr))
    release_coeff = np.exp(-1.0 / (0.220 * sr))
    env = np.zeros_like(out)
    curr = 0.0
    for i in range(len(out)):
        a = abs(out[i])
        if a > curr:
            curr = a + attack_coeff * (curr - a)
        else:
            curr = a + release_coeff * (curr - a)
        env[i] = curr

    thresh = 0.005
    floor = 0.001
    gate_gain = np.clip((env - floor) / (thresh - floor), 0.0, 1.0)
    gate_gain = 0.5 * (1.0 - np.cos(np.pi * gate_gain))
    gated = out * gate_gain

    # 4. Equal Loudness Normalization (Target active speech RMS: 0.125)
    target_rms = 0.125
    active_samples = gated[env > 0.01]
    if len(active_samples) > 0:
        active_rms = np.sqrt(np.mean(active_samples ** 2))
    else:
        active_rms = np.sqrt(np.mean(gated ** 2))

    if active_rms > 0.001:
        loudness_gain = target_rms / active_rms
        levelled = gated * loudness_gain
    else:
        levelled = gated

    # 5. Peak Limiter (-1 dB / 0.89)
    peak = np.max(np.abs(levelled))
    if peak > 0.89:
        levelled = levelled * (0.89 / peak)
    elif peak < 0.5 and peak > 0.001:
        levelled = levelled * (0.85 / peak)

    return levelled.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. RUNPOD SERVERLESS HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    job_input = job.get("input", job)
    raw_text = job_input.get("text", "").strip()
    if not raw_text:
        return {"error": "Matn kiritilmagan ('text' bo'sh)", "status": "FAILED"}

    voice = job_input.get("voice", "modern").lower()
    style = job_input.get("style", "adabiy").lower()
    speed = float(job_input.get("speed", 1.0))
    steps = int(job_input.get("steps", 32))
    fmt = job_input.get("format", "mp3").lower()
    seed = job_input.get("seed", 42)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    profile = VOICE_PROFILES.get(voice, VOICE_PROFILES["modern"])
    ref_audio = profile["audio"]
    ref_mel_len = profile["mel_len"]
    effective_speed = speed * profile.get("speed_factor", 1.0)
    
    r_text = clean_text_strictly_for_vocab(profile["ref_text"], vocab_char_map)

    # 1. Full Uzbek text normalization (numbers, dates, abbreviations, orthoepy, dialect, tutuq)
    norm_text = normalize_uzbek_text(raw_text, style=style).lower()
    norm_text = re.sub(r'\baudio', 'avdio', norm_text)
    norm_text = re.sub(r'\btts\b', 'te te es', norm_text)
    norm_text = re.sub(r'\bai\b', 'ey ay', norm_text)
    norm_text = re.sub(r'\bit\b', 'ay ti', norm_text)
    norm_text = re.sub(r'\bsentabr\b|\bsentyabr\b', 'sentiyabr', norm_text)
    norm_text = re.sub(r'\boktyabr\b', 'oktabr', norm_text)
    norm_text = re.sub(r'\bob[- ]havo\b', 'obhavo', norm_text)
    norm_text = re.sub(r'\bsoha', 'sohha', norm_text)
    norm_text = unicodedata.normalize('NFC', norm_text)
    norm_text = re.sub(r"[`'ʻʼʽ՚’‘]", "'", norm_text)
    norm_text = re.sub(r'\s+', ' ', norm_text).strip()

    # 2. Hierarchical clause splitting (max 200 chars)
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', norm_text) if s.strip()]
    final_raw_chunks = []
    for sent in raw_sentences:
        final_raw_chunks.extend(split_clause_smart(sent, max_chars=200))

    # 3. Identify pause types and clean strictly for vocab
    text_chunks = []
    chunk_pause_types = []
    for rc in final_raw_chunks:
        rc_clean = rc.strip()
        if re.search(r'[.!?]$', rc_clean):
            p_type = 'terminal'
        elif re.search(r'[,;:]$', rc_clean):
            p_type = 'clause'
        else:
            p_type = 'connector'

        rc_for_vocab = rc_clean.replace(',', '.')
        c = clean_text_strictly_for_vocab(rc_for_vocab, vocab_char_map, style=style, already_normalized=True)
        c = c.strip().strip(',;:').strip()
        if c:
            text_chunks.append(c)
            chunk_pause_types.append(p_type)

    if not text_chunks:
        return {"error": "Matn tozalangandan so'ng bo'sh qoldi", "status": "FAILED"}

    # 4. Inference loop per chunk with exact phonetic duration
    generated_waves = []
    for clean_chunk, p_type in zip(text_chunks, chunk_pause_types):
        is_terminal = (p_type == 'terminal')
        chunk_for_model = clean_chunk if clean_chunk.endswith(('.', '?', '!')) else (clean_chunk + ('.' if is_terminal else ','))

        dur_sec = calculate_phonetic_duration(chunk_for_model, is_terminal_sentence=is_terminal, speed_factor=effective_speed)
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
                seed=seed,
            )
            gen = gen.to(torch.float32)[:, ref_mel_len:, :].permute(0, 2, 1)
            mel_spec = gen.to(vocoder_device)
            wave_chunk = vocoder.decode(mel_spec).squeeze().cpu().numpy()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            generated_waves.append((wave_chunk, p_type))

    # 5. Studio-Grade Zero-Defect Stitching (Speech bounds cleaning + natural pauses)
    raw_chunks = [w for w, _ in generated_waves]
    p_types = [pt for _, pt in generated_waves]

    wave = stitch_chunks_zero_defect(
        raw_chunks,
        p_types,
        sr=target_sample_rate,
        terminal_pause_ms=260,
        clause_pause_ms=100,
        connector_pause_ms=80
    )

    # 6. Apply Studio DSP
    full_audio = apply_studio_dsp(wave, sr=target_sample_rate, profile=voice)
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
                "-qscale:a", "2", mp3_path
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
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
