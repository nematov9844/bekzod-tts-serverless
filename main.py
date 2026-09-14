"""
main.py — Bekzod Voice TTS RunPod Serverless Endpoint (140k F5-TTS)
"""

from runpod_flash import Endpoint, GpuGroup

tts_app = Endpoint(
    name="bekzod-tts-serverless",
    gpu=GpuGroup.ANY,
    workers=(0, 2),
    idle_timeout=120,
    system_dependencies=["ffmpeg"],
    dependencies=[
        "vocos",
        "scipy",
        "huggingface-hub",
        "safetensors",
        "torchaudio",
        "soundfile",
    ],
)


@tts_app
async def synthesize(**kwargs) -> dict:
    """
    Bekzod Voice TTS Serverless Handler.
    Supports both {"input": {"text": "..."}} and direct {"text": "..."}.
    """
    import os
    import io
    import re
    import base64
    import tempfile
    import subprocess
    import unicodedata
    from pathlib import Path

    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np
    import scipy.signal as signal
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    from f5_tts.model import CFM, DiT
    from f5_tts.infer.utils_infer import load_vocoder, target_sample_rate, hop_length

    global _BEKZOD_ENGINE
    try:
        engine = _BEKZOD_ENGINE
    except NameError:
        print("[*] Worker initializatsiya: HuggingFace'dan model va ankorlarni yuklash...")
        repo_id = "lynx9844/f5tts-bekzod-200k-uzbek"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Vocab yuklash
        vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.txt")
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = [line.strip("\n") for line in f]
        valid_vocab = [l for i, l in enumerate(raw_vocab) if l != "" or i == 0]
        vocab_char_map = {l: i for i, l in enumerate(valid_vocab)}

        # 2. Checkpoint yuklash
        ckpt_name = os.environ.get("MODEL_CHECKPOINT", "model_150000_studio_clean.safetensors")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)

        # 3. Modelni qurish
        model = DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_num_embeds=len(vocab_char_map),
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
        if device == "cuda":
            cfm = cfm.half().to(device).eval()
        else:
            cfm = cfm.to(device).eval()

        vocoder = load_vocoder("vocos", device="cpu")

        # 4. Ankor audiolarni yuklash
        def load_anchor(filename: str):
            p = hf_hub_download(repo_id=repo_id, filename=filename)
            a, sr = torchaudio.load(p)
            if sr != target_sample_rate:
                a = torchaudio.functional.resample(a, sr, target_sample_rate)
            if a.shape[0] > 1:
                a = torch.mean(a, dim=0, keepdim=True)
            if device == "cuda":
                a = a.half().to(device)
            return a, a.shape[-1] // hop_length

        modern_audio, modern_len = load_anchor("ref_modern_active.wav")
        classic_audio, classic_len = load_anchor("ref_classic_baritone.wav")

        _BEKZOD_ENGINE = {
            "cfm": cfm,
            "vocoder": vocoder,
            "vocab_char_map": vocab_char_map,
            "device": device,
            "modern_audio": modern_audio,
            "modern_len": modern_len,
            "modern_text": "transport orqali yevropada yevropa portiga u yerdan temir yo'l.",
            "classic_audio": classic_audio,
            "classic_len": classic_len,
            "classic_text": "asosiy qismlari yaponiyada ishlab chiqarilgan elektronikasi janubiy koreyada tayyorlangan.",
        }
        engine = _BEKZOD_ENGINE
        print("[✓] Bekzod TTS Engine muvaffaqiyatli ishga tushdi!")

    # ─── Parametrlarni ajratib olish ─────────────────────────────────────────
    payload = kwargs.get("input", kwargs)
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "Matn bo'sh bo'lishi mumkin emas", "status": "FAILED"}

    voice = payload.get("voice", "modern").lower()
    speed = float(payload.get("speed", 1.0))
    fmt = payload.get("format", "mp3").lower()
    seed = payload.get("seed", 42)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    # ─── Fonetik hisoblagich ────────────────────────────────────────────────
    PHONEME_DURATIONS_MS = {
        'a': 110, 'o': 115, 'e': 105, 'i': 70, 'u': 75, "o'": 130,
        's': 90, 'z': 85, 'sh': 100, 'ch': 95, 'x': 95, 'h': 90, 'f': 85,
        'p': 60, 't': 60, 'k': 65, 'q': 70, 'b': 60, 'd': 60, 'g': 65,
        'm': 80, 'n': 80, 'l': 75, 'r': 80, 'y': 75, 'v': 75, 'j': 80,
        "'": 65, ' ': 50
    }
    VOWELS = set("aoeiu")

    def calc_duration(t: str, spd: float = 1.0) -> float:
        clean_t = t.lower()
        clean_t = unicodedata.normalize("NFC", clean_t)
        clean_t = re.sub(r"[`'ʻʼʽ՚’‘]", "'", clean_t)
        words = clean_t.split()
        if not words:
            return 2.0
        total_ms = 0.0
        for w_idx, word in enumerate(words):
            is_last = (w_idx == len(words) - 1)
            i = 0
            w_len = len(word)
            while i < w_len:
                if i + 2 <= w_len and word[i:i+2] in ["o'", "g'", "sh", "ch"]:
                    dur = PHONEME_DURATIONS_MS.get(word[i:i+2], 95)
                    total_ms += dur
                    i += 2
                    continue
                ch = word[i]
                dur = PHONEME_DURATIONS_MS.get(ch, 75)
                if i + 1 < w_len and word[i+1] == ch and ch not in VOWELS:
                    dur += 45
                if i + 1 < w_len and ch in VOWELS and word[i+1] in VOWELS and word[i+1] != "'":
                    dur += 45
                total_ms += dur
                i += 1
            total_ms += PHONEME_DURATIONS_MS[' ']
            if is_last:
                total_ms += 160.0
        return max(2.0, (total_ms / 1000.0) / spd)

    def clean_u(t: str) -> str:
        t = unicodedata.normalize('NFC', t)
        t = re.sub(r'[\u2018\u2019\u02BB\u02BC\`]', "'", t)
        t = re.sub(r"[^a-z0-9\s.,!?\'\-]", ' ', t.lower())
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def apply_dsp(wave: np.ndarray, prof: str) -> np.ndarray:
        out = wave.copy()
        sos_hp = signal.butter(2, 70, 'hp', fs=target_sample_rate, output='sos')
        out = signal.sosfiltfilt(sos_hp, out)
        if prof == "classic":
            gain = 10 ** (1.8 / 20.0)
            sos_low = signal.butter(2, 150, 'lp', fs=target_sample_rate, output='sos')
            low_band = signal.sosfiltfilt(sos_low, out)
            out = out + (gain - 1.0) * low_band
        else:
            sos_lp = signal.butter(2, 8500, 'lp', fs=target_sample_rate, output='sos')
            out = signal.sosfiltfilt(sos_lp, out)
        peak = np.max(np.abs(out))
        if peak > 1e-5:
            out = out * (0.89 / peak)
        return out.astype(np.float32)

    # Profil tanlash
    if voice == "classic":
        ref_audio = engine["classic_audio"]
        ref_audio_len = engine["classic_len"]
        ref_clean = clean_u(engine["classic_text"])
        effective_speed = speed * 0.98
    else:
        ref_audio = engine["modern_audio"]
        ref_audio_len = engine["modern_len"]
        ref_clean = clean_u(engine["modern_text"])
        effective_speed = speed * 1.05

    # Gaplarni ajratish
    raw_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    sentences = []
    for sent in raw_sents:
        if len(sent) <= 200:
            sentences.append(sent)
        else:
            parts = [p.strip() for p in re.split(r'(?<=[,;:])\s+', sent) if p.strip()]
            sentences.extend(parts if parts else [sent])

    waves = []
    pause_samples = int(0.24 * target_sample_rate)

    for idx, sent in enumerate(sentences, 1):
        c_sent = clean_u(sent)
        full_text = [ref_clean + " " + c_sent]

        dur_sec = calc_duration(c_sent, spd=effective_speed)
        target_frames = int(dur_sec * target_sample_rate / hop_length)
        duration = ref_audio_len + target_frames

        with torch.inference_mode():
            gen, _ = engine["cfm"].sample(
                cond=ref_audio,
                text=full_text,
                duration=duration,
                steps=32,
                cfg_strength=1.55,
                sway_sampling_coef=-1.0,
                seed=seed + idx if seed is not None else None,
            )
            gen = gen.to(torch.float32)[:, ref_audio_len:, :].permute(0, 2, 1).cpu()
            w = engine["vocoder"].decode(gen).squeeze().cpu().numpy()

            fade = int(0.015 * target_sample_rate)
            if len(w) > fade:
                w[-fade:] *= np.linspace(1, 0, fade)

            waves.append(w)
            if idx < len(sentences):
                waves.append(np.zeros(pause_samples, dtype=np.float32))

            if engine["device"] == "cuda":
                torch.cuda.empty_cache()

    full_audio = np.concatenate(waves)
    full_audio = apply_dsp(full_audio, prof=voice)
    total_duration = round(len(full_audio) / target_sample_rate, 2)

    # ─── Audio formatga o'tkazish (WAV yoki MP3) ──────────────────────────────
    if fmt == "mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, full_audio, target_sample_rate)
            wav_path = tmp_wav.name
        mp3_path = wav_path.replace(".wav", ".mp3")
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame",
            "-qscale:a", "2", mp3_path
        ], capture_output=True, check=True)
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()
        os.unlink(wav_path)
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
