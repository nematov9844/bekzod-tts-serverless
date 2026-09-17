"""
handler.py — RunPod Serverless Worker for Bekzod Voice 150k F5-TTS
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
from huggingface_hub import hf_hub_download, HfApi, create_repo
import runpod
import google.generativeai as genai

from f5_tts.model import CFM, DiT
from f5_tts.infer.utils_infer import load_vocoder, target_sample_rate, hop_length

# Local modules
from text_normalizer import normalize_uzbek_text
from phonetic_engine import calculate_phonetic_duration
from audio_stitcher import clean_speech_bounds
from cyrillic_to_latin import cyrillic_to_latin

# ─────────────────────────────────────────────────────────────────────────────
# 1. INITIALIZATION & ASSET LOADING (RUNS ONCE ON WORKER COLD START)
# ─────────────────────────────────────────────────────────────────────────────

print("[*] Initializing Bekzod TTS 150k Studio Clean Engine on cold start...")

REPO_ID = os.environ.get("HF_REPO_ID", "lynx9844/f5tts-bekzod-200k-uzbek")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RunPod's /run endpoint caps responses at 10MB (/runsync at 20MB); base64
# adds ~33% overhead, so anything much over ~6MB raw audio risks the job
# completing successfully server-side while RunPod silently drops the
# response (confirmed live: a 4000-word document took ~7-8 minutes to
# synthesize and returned COMPLETED with a completely empty output, in
# both wav and mp3). Long documents get uploaded to a public HF dataset
# repo instead, and the response carries a URL rather than inline bytes.
OUTPUT_SIZE_LIMIT_BYTES = 6_000_000
OUTPUT_REPO = os.environ.get("HF_OUTPUT_REPO", "lynx9844/bekzod-tts-outputs")
_output_repo_ready = False


def _ensure_output_repo():
    global _output_repo_ready
    if _output_repo_ready:
        return
    try:
        create_repo(OUTPUT_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True, private=False)
    except Exception as e:
        print(f"[!] Could not ensure HF output repo {OUTPUT_REPO}: {e}")
    _output_repo_ready = True


def upload_large_output(audio_bytes: bytes, fmt: str) -> str:
    """Uploads audio too large for RunPod's inline payload limit to a public
    HF dataset repo and returns a direct download URL."""
    import uuid
    _ensure_output_repo()
    fname = f"{uuid.uuid4().hex}.{fmt}"
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=io.BytesIO(audio_bytes),
        path_in_repo=fname,
        repo_id=OUTPUT_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    return f"https://huggingface.co/datasets/{OUTPUT_REPO}/resolve/main/{fname}"
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

PROOFREAD_PROMPT = """You are a strict Uzbek Latin-script orthography proofreader.
Fix ONLY spelling, punctuation, and obvious text-extraction/OCR artifacts
(broken words, wrong apostrophes, stray characters) in the text below.
Do NOT add, remove, summarize, rephrase, or reorder any content. Do NOT
change the subject matter or meaning. Preserve paragraph breaks. Return
ONLY the corrected text, with no commentary, no markdown fences, nothing else.

TEXT:
{text}
"""


def proofread_uzbek_text(text: str) -> str:
    """Fixes spelling/orthography only via Gemini, content-preserving. Falls
    back to the original text untouched if Gemini is unavailable or fails --
    this is a proofreading pass, not a required step."""
    if not GEMINI_API_KEY or not text.strip():
        return text
    available_models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
    for model_name in available_models:
        try:
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(
                PROOFREAD_PROMPT.format(text=text),
                request_options={"timeout": 60.0},
            )
            if response and response.text and response.text.strip():
                out = response.text.strip()
                if out.startswith("```"):
                    lines = out.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    out = "\n".join(lines).strip()
                return out
        except Exception as e:
            print(f"[!] Gemini proofread model {model_name} failed: {e}")
    print("[!] Gemini proofreading unavailable, using original text")
    return text

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

# 2. Checkpoint (model_150000_studio_clean.safetensors)
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "model_150000_studio_clean.safetensors")
ckpt_path = get_file(MODEL_CHECKPOINT)

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

# 4. Reference Anchors (Clean Studio-Grade)
def load_anchor(filename: str, fallback: Optional[str] = None):
    try:
        p = get_file(filename)
    except Exception as e:
        if fallback:
            print(f"[!] {filename} topilmadi, fallback {fallback} ishlatilmoqda: {e}")
            p = get_file(fallback)
        else:
            raise e
    a, sr = torchaudio.load(p)
    if sr != target_sample_rate:
        a = torchaudio.functional.resample(a, sr, target_sample_rate)
    if a.shape[0] > 1:
        a = torch.mean(a, dim=0, keepdim=True)
    if DEVICE == "cuda":
        a = a.half().to(DEVICE)
    mel_len = a.shape[-1] // hop_length
    return a, mel_len

classic_audio, classic_len = load_anchor("clean_ref_classic_baritone.wav", fallback="ref_classic_baritone.wav")
modern_audio, modern_len = load_anchor("clean_ref_modern_active.wav", fallback="ref_modern_active.wav")
storyteller_audio, storyteller_len = load_anchor("ref_storyteller_v4_clean.wav", fallback="clean_ref_classic_baritone.wav")
inquisitive_audio, inquisitive_len = load_anchor("ref_inquisitive_v4_clean.wav", fallback="clean_ref_modern_active.wav")
cheerful_audio, cheerful_len = load_anchor("ref_cheerful_v1_clean.wav", fallback="clean_ref_modern_active.wav")
melancholic_audio, melancholic_len = load_anchor("ref_melancholic_v5_pure.wav", fallback="clean_ref_classic_baritone.wav")
epic_audio, epic_len = load_anchor("ref_epic_v1_pure.wav", fallback="clean_ref_modern_active.wav")
mysterious_audio, mysterious_len = load_anchor("ref_mysterious_v1_pure.wav", fallback="clean_ref_classic_baritone.wav")
authoritative_audio, authoritative_len = load_anchor("ref_authoritative_v1_pure.wav", fallback="clean_ref_classic_baritone.wav")
ironic_audio, ironic_len = load_anchor("ref_ironic_v1_pure.wav", fallback="clean_ref_modern_active.wav")

VOICE_PROFILES = {
    "classic": {
        "audio": classic_audio,
        "ref_text": "asosiy qismlari yaponiyada ishlab chiqarilgan elektronikasi janubiy koreyada tayyorlangan.",
        "mel_len": classic_len,
        "speed_factor": 1.0,
        "pause_ms": 0.24,
        "cfg_strength": 1.55,
    },
    "modern": {
        "audio": modern_audio,
        "ref_text": "transport orqali yevropada yevropa portiga u yerdan temir yo'l.",
        "mel_len": modern_len,
        "speed_factor": 1.02,
        "pause_ms": 0.20,
        "cfg_strength": 1.55,
    },
    "storyteller": {
        "audio": storyteller_audio,
        "ref_text": "tasavvur qiling.",
        "mel_len": storyteller_len,
        "speed_factor": 0.96,
        "pause_ms": 0.34,
        "cfg_strength": 1.50,
    },
    "inquisitive": {
        "audio": inquisitive_audio,
        "ref_text": "bu savol amaliyotda juda qiziqtiradi.",
        "mel_len": inquisitive_len,
        "speed_factor": 1.02,
        "pause_ms": 0.22,
        "cfg_strength": 1.60,
    },
    "cheerful": {
        "audio": cheerful_audio,
        "ref_text": "tashqi iqtisodiy faoliyatda qonuniy raqobatni ta'minlaydi.",
        "mel_len": cheerful_len,
        "speed_factor": 1.05,
        "pause_ms": 0.18,
        "cfg_strength": 1.70,
    },
    "melancholic": {
        "audio": melancholic_audio,
        "ref_text": "tashqi savdoni soddalashtirishga xizmat qilmoqda.",
        "mel_len": melancholic_len,
        "speed_factor": 0.88,
        "pause_ms": 0.45,
        "cfg_strength": 1.45,
    },
    "epic": {
        "audio": epic_audio,
        "ref_text": "shuning uchun bojxona nazorati davlatning iqtisodiy xavfsizligini ta'minlovchi muhim vositalardan biri hisoblanadi.",
        "mel_len": epic_len,
        "speed_factor": 0.96,
        "pause_ms": 0.30,
        "cfg_strength": 1.60,
    },
    "mysterious": {
        "audio": mysterious_audio,
        "ref_text": "konvensiyaning asosiy maqsadi etib bojxona tartib taomillarini soddalashtirish.",
        "mel_len": mysterious_len,
        "speed_factor": 0.91,
        "pause_ms": 0.42,
        "cfg_strength": 1.45,
    },
    "authoritative": {
        "audio": authoritative_audio,
        "ref_text": "yo'q ayrim harakatlarga qonunchilikda ruxsat berilgan.",
        "mel_len": authoritative_len,
        "speed_factor": 0.98,
        "pause_ms": 0.20,
        "cfg_strength": 1.65,
    },
    "ironic": {
        "audio": ironic_audio,
        "ref_text": "nega bu qadar ko'p talablar bor bu talablar davlat tomonidan shunchaki o'rnatilmagan.",
        "mel_len": ironic_len,
        "speed_factor": 0.97,
        "pause_ms": 0.28,
        "cfg_strength": 1.55,
    }
}
# Uzbek and semantic aliases
VOICE_PROFILES["vazmin"] = VOICE_PROFILES["classic"]
VOICE_PROFILES["podkast"] = VOICE_PROFILES["modern"]
VOICE_PROFILES["ertakchi"] = VOICE_PROFILES["storyteller"]
VOICE_PROFILES["savol"] = VOICE_PROFILES["inquisitive"]
VOICE_PROFILES["quvnoq"] = VOICE_PROFILES["cheerful"]
VOICE_PROFILES["shodiyona"] = VOICE_PROFILES["cheerful"]
VOICE_PROFILES["gamgin"] = VOICE_PROFILES["melancholic"]
VOICE_PROFILES["mayus"] = VOICE_PROFILES["melancholic"]
VOICE_PROFILES["dramatic"] = VOICE_PROFILES["melancholic"]
VOICE_PROFILES["tantanavor"] = VOICE_PROFILES["epic"]
VOICE_PROFILES["shijoatli"] = VOICE_PROFILES["epic"]
VOICE_PROFILES["sirli"] = VOICE_PROFILES["mysterious"]
VOICE_PROFILES["pinhona"] = VOICE_PROFILES["mysterious"]
VOICE_PROFILES["qatiy"] = VOICE_PROFILES["authoritative"]
VOICE_PROFILES["qat'iy"] = VOICE_PROFILES["authoritative"]
VOICE_PROFILES["buyruq"] = VOICE_PROFILES["authoritative"]
VOICE_PROFILES["kinoyali"] = VOICE_PROFILES["ironic"]
VOICE_PROFILES["sarkazm"] = VOICE_PROFILES["ironic"]

print(f"[✓] Bekzod TTS Engine (10 Expressive Styles Matrix) initialized successfully on {DEVICE}!")

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
    
    # Phonetic fix for months, loanwords and hiatus (io/ia/ie -> yo/ya/ye)
    norm_text = re.sub(r'\baudio', 'audyo', norm_text)
    norm_text = re.sub(r'\bvideo', 'vidyo', norm_text)
    norm_text = re.sub(r'\bradio', 'radyo', norm_text)
    norm_text = re.sub(r'\bstudio', 'studyo', norm_text)
    norm_text = re.sub(r'\bstadion', 'stadyon', norm_text)
    norm_text = re.sub(r'\bchempion', 'chempyon', norm_text)
    norm_text = re.sub(r'\bregion', 'regyon', norm_text)
    norm_text = re.sub(r'\bmillion', 'milyon', norm_text)
    norm_text = re.sub(r'\bmilliard', 'milyard', norm_text)
    norm_text = re.sub(r'\bbilliard', 'bilyard', norm_text)
    norm_text = re.sub(r'\bmaterial', 'materyal', norm_text)
    norm_text = re.sub(r'\bvariant', 'varyant', norm_text)
    norm_text = re.sub(r'\baviatsiy', 'avyatsiy', norm_text)
    norm_text = re.sub(r'\baviasiy', 'avyatsiy', norm_text)
    norm_text = re.sub(r'\bpianino', 'pyanino', norm_text)
    norm_text = re.sub(r'\bsentabr\b|\bsentyabr\b', 'sentiyabr', norm_text)
    norm_text = re.sub(r'\boktyabr\b', 'oktabr', norm_text)
    norm_text = re.sub(r'\bob[- ]havo\b', 'obhavo', norm_text)
    norm_text = re.sub(r'\bsoha', 'sohha', norm_text)

    # Hiatus reinforcement
    norm_text = re.sub(r'\boila', 'oiila', norm_text)
    norm_text = re.sub(r'\bdoira', 'doiira', norm_text)
    norm_text = re.sub(r'\bshoir', 'shoiir', norm_text)
    norm_text = re.sub(r'\brais\b', 'raiis', norm_text)
    norm_text = re.sub(r'\bfoiz\b', 'foiiz', norm_text)

    # Raqamlar va yuzliklar ritmik birikishi (to'rtyuz, beshyuz, uchyuz)
    norm_text = re.sub(r'\b(bir|ikki|uch|to\'rt|besh|olti|yetti|sakkiz|to\'qqiz)\s+yuz\b', r'\1yuz', norm_text)

    # Kontakt va progressiv/regressiv fonetik assimilatsiya (shamba, tussiz, kitopka)
    norm_text = re.sub(r'\bshanba\b', 'shamba', norm_text)
    norm_text = re.sub(r'\bdushanba\b', 'dushamba', norm_text)
    norm_text = re.sub(r'\bseshanba\b', 'seshamba', norm_text)
    norm_text = re.sub(r'\bchorshanba\b', 'chorshamba', norm_text)
    norm_text = re.sub(r'\bpayshanba\b', 'payshamba', norm_text)
    norm_text = re.sub(r'\bmanba\b', 'mamba', norm_text)
    norm_text = re.sub(r'\byonbosh\b', 'yombosh', norm_text)
    norm_text = re.sub(r'\btuzsiz', 'tussiz', norm_text)
    norm_text = re.sub(r'\b(ayt|ot|ket|tut|kut|yot)gan\b', r'\1kan', norm_text)
    norm_text = re.sub(r'\b(kitob|maktab|hisob|sabab|asbob)ga\b', r'\1ka', norm_text)
    norm_text = re.sub(r'\buchta\b', 'ushta', norm_text)
    norm_text = re.sub(r'\byuzta\b', 'yusta', norm_text)

    # Dashes to spaces
    norm_text = re.sub(r'[-–—_]+', ' ', norm_text)

    # Compound words breakdown
    norm_text = re.sub(r'\bneyrotarmoq', 'neyro tarmoq', norm_text)
    norm_text = re.sub(r'\bneyrobiolog', 'neyro biolog', norm_text)
    norm_text = re.sub(r'\bnanotexnolog', 'nano texnolog', norm_text)
    norm_text = re.sub(r'\bbiotibbiyot', 'bio tibbiyot', norm_text)
    norm_text = re.sub(r'\bbiotexnolog', 'bio texnolog', norm_text)
    norm_text = re.sub(r'\bkiberxavfsiz', 'kiber xavfsiz', norm_text)
    norm_text = re.sub(r'\belektromobil', 'elektro mobil', norm_text)
    norm_text = re.sub(r'\binfratuzilma', 'infra tuzilma', norm_text)
    norm_text = re.sub(r'\btelekommunikatsiya', 'tele kommunikatsiya', norm_text)
    norm_text = re.sub(r'\bgidroelektr', 'gidro elektr', norm_text)
    norm_text = re.sub(r'\bavtotransport', 'avto transport', norm_text)
    norm_text = re.sub(r'\bmikroiqtisod', 'mikro iqtisod', norm_text)
    norm_text = re.sub(r'\bmakroiqtisod', 'makro iqtisod', norm_text)
    norm_text = re.sub(r'\bsuperkompyuter', 'super kompyuter', norm_text)
    norm_text = re.sub(r'\bekotizim', 'eko tizim', norm_text)
    norm_text = re.sub(r'\bumumta\'lim', 'umum ta\'lim', norm_text)
    norm_text = re.sub(r'\bsun\'iyintellekt', 'sun\'iy intellekt', norm_text)
    norm_text = re.sub(r'\baudiokitob', 'audyo kitob', norm_text)
    norm_text = re.sub(r'\bvideodars', 'vidyo dars', norm_text)
    norm_text = re.sub(r'\bvideorolik', 'vidyo rolik', norm_text)
    norm_text = re.sub(r'\bvideokonferensiya', 'vidyo konferensiya', norm_text)
    norm_text = re.sub(r'\bvebsayt', 'veb sayt', norm_text)

    # Number suffixes attachment
    norm_text = re.sub(r'\b(bir|ikki|uch|to\'rt|besh|olti|yetti|sakkiz|to\'qqiz|o\'n|yigirma|o\'ttiz|qirq|ellik|oltmish|yetmish|sakson|sakkson|to\'qson|yuz|ming|million|milliyon|milliard)\s+(dan|ga|da|ni|ning)\b', r'\1\2', norm_text)

    # Pronoun and clitic reductions (hech kim -> hechkim, hech narsa -> hechnarsa)
    norm_text = re.sub(r'\bhech\s+kim\b', 'hechkim', norm_text)
    norm_text = re.sub(r'\bhech\s+narsa\b', 'hechnarsa', norm_text)
    norm_text = re.sub(r'\bhech\s+qachon\b', 'hechqachon', norm_text)
    norm_text = re.sub(r'\bhech\s+qanday\b', 'hechqanday', norm_text)
    norm_text = re.sub(r'\bhech\s+qayer\b', 'hechqayer', norm_text)
    norm_text = re.sub(r'\bhech\s+biri\b', 'hechbiri', norm_text)
    norm_text = re.sub(r'\bhar\s+bir\b', 'harbir', norm_text)
    norm_text = re.sub(r'\bhar\s+doim\b', 'hardoyim', norm_text)
    norm_text = re.sub(r'\bhar\s+kim\b', 'harkim', norm_text)
    norm_text = re.sub(r'\bhar\s+qanday\b', 'harqanday', norm_text)
    norm_text = re.sub(r'\bbiror\s+bir\b', 'birorbir', norm_text)

    # Narrative clitics reduction (bor ekan -> borekan, yo\'q ekan -> yo\'qekan)
    norm_text = re.sub(r'\bbor\s+ekan\b', 'borekan', norm_text)
    norm_text = re.sub(r'\byo\'q\s+ekan\b', 'yo\'qekan', norm_text)
    norm_text = re.sub(r'\bbo\'lgan\s+ekan\b', 'bo\'lganekan', norm_text)
    norm_text = re.sub(r'\bsezmagan\s+ekan\b', 'sezmaganekan', norm_text)
    norm_text = re.sub(r'-ku\b', 'ku', norm_text)

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

def split_sentences_natural(text: str, max_chars: int = 100, word_max_chars: int = 70) -> List[str]:
    """
    Splits text strictly by sentence boundaries (.!? or newlines \n).
    If any block > max_chars, splits by clauses (, ; :).
    If still > max_chars (e.g. unpunctuated wall of text), splits by words with word_max_chars!
    Guarantees no single chunk ever exceeds safe diffusion horizon.
    """
    raw_blocks = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    if not raw_blocks:
        raw_blocks = [text.strip()]

    chunks = []
    for block in raw_blocks:
        if len(block) <= max_chars:
            chunks.append(block)
            continue

        clauses = [c.strip() for c in re.split(r'(?<=[,;:])\s+', block) if c.strip()]
        for clause in clauses:
            if len(clause) <= max_chars:
                chunks.append(clause)
                continue

            # Fail-safe word splitting for long clauses without punctuation
            words = clause.split()
            current = ""
            for w in words:
                if len(current) + len(w) + 1 <= word_max_chars:
                    current = f"{current} {w}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = w
            if current:
                chunks.append(current)

    return chunks

def biquad_peak(wave: np.ndarray, fs: int, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """Standard Audio EQ Cookbook Peaking Filter (transparent bell EQ)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([a0, a1, a2], dtype=np.float64) / a0
    return signal.lfilter(b, a, wave.astype(np.float64)).astype(np.float32)

def biquad_highshelf(wave: np.ndarray, fs: int, freq: float, gain_db: float) -> np.ndarray:
    """Standard Audio EQ Cookbook High Shelf Filter (gentle air polish / softening)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
    cos_w0 = np.cos(w0)
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([a0, a1, a2], dtype=np.float64) / a0
    return signal.lfilter(b, a, wave.astype(np.float64)).astype(np.float32)

def apply_style_dsp(wave: np.ndarray, voice: str = "classic", sr: int = 24000) -> np.ndarray:
    """
    Studio-Grade Transparent Mastering EQ Engine
    Preserves 100% full-bandwidth speech fidelity (up to 12 kHz), crystal clear consonants, and zero muffling.
    """
    out = wave.copy().astype(np.float32)

    # 1. Clean sub-bass rumble (< 65 Hz) for all styles
    sos_hp = signal.butter(2, 65, 'hp', fs=sr, output='sos')
    out = signal.sosfiltfilt(sos_hp, out)

    if voice in ["melancholic", "gamgin", "mayus", "dramatic"]:
        # Warm chest sadness (+1.2 dB @ 200 Hz) + gentle top softening (-1.5 dB shelf @ 7000 Hz, zero lowpass)
        out = biquad_peak(out, sr, 200, 1.2, q=1.0)
        out = biquad_highshelf(out, sr, 7000, -1.5)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.82 / peak)

    elif voice in ["mysterious", "sirli", "pinhona"]:
        # Intimate proximity (+1.0 dB @ 180 Hz) + gentle air softening (-1.8 dB shelf @ 6500 Hz)
        out = biquad_peak(out, sr, 180, 1.0, q=1.0)
        out = biquad_highshelf(out, sr, 6500, -1.8)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.78 / peak)

    elif voice in ["epic", "tantanavor", "shijoatli"]:
        # Resonant chest power (+1.5 dB @ 200 Hz) + crisp brilliance (+1.8 dB @ 3500 Hz)
        out = biquad_peak(out, sr, 200, 1.5, q=1.0)
        out = biquad_peak(out, sr, 3500, 1.8, q=1.2)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.95 / peak)

    elif voice in ["cheerful", "quvnoq", "shodiyona"]:
        # Smiling clarity (+1.8 dB @ 3600 Hz)
        out = biquad_peak(out, sr, 3600, 1.8, q=1.0)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.92 / peak)

    elif voice in ["authoritative", "qatiy", "qat'iy", "buyruq"]:
        # Sharp articulation (+2.0 dB @ 2800 Hz)
        out = biquad_peak(out, sr, 2800, 2.0, q=1.1)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.92 / peak)

    elif voice in ["storyteller", "ertakchi"]:
        # Velvet warmth (+1.5 dB @ 220 Hz)
        out = biquad_peak(out, sr, 220, 1.5, q=1.0)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.86 / peak)

    elif voice in ["ironic", "kinoyali", "sarkazm"]:
        # Smirking presence (+1.4 dB @ 3200 Hz)
        out = biquad_peak(out, sr, 3200, 1.4, q=1.0)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.88 / peak)

    elif voice in ["modern", "podkast"]:
        # Punchy podcast presence (+1.2 dB @ 2600 Hz)
        out = biquad_peak(out, sr, 2600, 1.2, q=1.0)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.90 / peak)

    elif voice in ["inquisitive", "savol"]:
        # Question clarity (+1.4 dB @ 3000 Hz)
        out = biquad_peak(out, sr, 3000, 1.4, q=1.0)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.88 / peak)

    else:
        # Classic / Vazmin: natural balanced baritone (+1.2 dB @ 180 Hz)
        out = biquad_peak(out, sr, 180, 1.2, q=1.0)
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
    # Safety net: convert any Cyrillic Uzbek text to Latin before normalization
    # (frontend already does this client-side, but a direct API caller might
    # send raw Cyrillic). No-op on already-Latin text.
    raw_text = cyrillic_to_latin(raw_text)

    # Optional Gemini spelling/orthography proofread pass (e.g. for text
    # extracted from uploaded PDF/DOC/XLS files, which often has OCR/
    # extraction artifacts). Content-preserving -- fixes spelling only.
    if job_input.get("proofread"):
        raw_text = proofread_uzbek_text(raw_text)

    voice = (job_input.get("voice") or job_input.get("voice_style") or job_input.get("style_name") or "classic").lower()
    style = job_input.get("style", "adabiy").lower()
    speed = float(job_input.get("speed", 1.0))
    steps = int(job_input.get("steps", 32))
    fmt = job_input.get("format", "mp3").lower()
    seed = job_input.get("seed", None)
    # "clean" (default True = shovqinsiz): applies the full cleanup pipeline
    # (per-chunk vocoder-bounds trimming + final mastering EQ). Set false
    # (shovqinli) to get the raw, unprocessed vocoder output instead.
    clean_mode = bool(job_input.get("clean", True))

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    profile = VOICE_PROFILES.get(voice, VOICE_PROFILES["classic"])
    ref_audio = profile["audio"]
    ref_mel_len = profile["mel_len"]
    effective_speed = speed * profile.get("speed_factor", 1.0)
    effective_cfg = float(job_input.get("cfg_strength", profile.get("cfg_strength", 1.55)))
    pause_ms = float(job_input.get("pause_duration", profile.get("pause_ms", 0.24)))
    
    r_text = clean_text_strictly_for_vocab(profile["ref_text"], vocab_char_map)

    # 1. Full Uzbek text normalization (numbers, dates, abbreviations, orthoepy, dialect, tutuq)
    norm_text = normalize_uzbek_text(raw_text, style=style).lower()
    norm_text = re.sub(r'\baudio\b', 'audyo', norm_text)
    norm_text = re.sub(r'\bvideo\b', 'vidyo', norm_text)
    norm_text = re.sub(r'\bradio\b', 'radyo', norm_text)
    norm_text = re.sub(r'\btts\b', 'te te es', norm_text)
    norm_text = re.sub(r'\bai\b', 'ey ay', norm_text)

    # 2. Sentence Splitting strictly by sentence boundaries (keeps commas inside clauses, safe word chunks <= 70 chars)
    raw_sentences = split_sentences_natural(norm_text, max_chars=100, word_max_chars=70)
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
                cfg_strength=effective_cfg,
                sway_sampling_coef=-1.0,
                seed=seed if seed is not None else (200 + idx)
            )
            gen = gen.to(torch.float32)[:, ref_mel_len:, :].permute(0, 2, 1)
            mel_spec = gen.to(vocoder_device)
            wave_chunk = vocoder.decode(mel_spec).squeeze().cpu().numpy()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            if clean_mode:
                # Clean vocoder onset latency and trailing vocoder air (70ms lead preserves initial plosives B, P)
                chunk_clean = clean_speech_bounds(
                    wave_chunk.astype(np.float32),
                    sr=target_sample_rate,
                    pad_lead_ms=70,
                    pad_tail_ms=160
                )
            else:
                chunk_clean = wave_chunk.astype(np.float32)
            generated_waves.append(chunk_clean)

    # 5. Natural human breath pause between sentences shaped per style
    pause_samples = int(pause_ms * target_sample_rate)
    final_pieces = []
    for idx, c in enumerate(generated_waves):
        final_pieces.append(c)
        if idx < len(generated_waves) - 1:
            final_pieces.append(np.zeros(pause_samples, dtype=np.float32))

    full_audio = np.concatenate(final_pieces)

    # 6. Apply Style Master DSP (energy, resonance, warmth, spectral contour per style)
    if clean_mode:
        full_audio = apply_style_dsp(full_audio, voice=voice, sr=target_sample_rate)
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

    if len(audio_bytes) > OUTPUT_SIZE_LIMIT_BYTES:
        try:
            audio_url = upload_large_output(audio_bytes, fmt)
            return {
                "status": "COMPLETED",
                "audio_url": audio_url,
                "duration": total_duration,
                "format": fmt,
                "voice": voice,
                "sample_rate": target_sample_rate,
                "steps": steps,
                "clean": clean_mode
            }
        except Exception as e:
            print(f"[!] Large-output HF upload failed ({e}), falling back to inline base64 "
                  f"-- this will likely exceed RunPod's payload limit and return empty.")

    b64_str = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "status": "COMPLETED",
        "audio_base64": b64_str,
        "duration": total_duration,
        "format": fmt,
        "voice": voice,
        "sample_rate": target_sample_rate,
        "steps": steps,
        "clean": clean_mode
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
