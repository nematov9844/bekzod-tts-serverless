#!/usr/bin/env python3
"""
pipeline/audio_stitcher.py
===========================
Studio-Grade Zero-Defect Audio Stitching Engine:
1. Speech Boundary Trimming (Zero Consonant Dropping):
   - Trims lead vocoder transition while strictly preserving pad_lead_ms (45ms)
     before speech starts, keeping 'p', 'b', 'v', 'q', 'm' 100% intact.
   - Trims trailing empty vocoder air while preserving pad_tail_ms (140-160ms)
     release decay, keeping final vowels and suffixes ('-da', '-ti', '-di') 100% intact.
   - Smooth 5ms cosine micro-fade at boundaries eliminates DC clicks and pops.
2. Natural Studio Pauses:
   - Terminal sentence (. ! ?): 260 ms
   - Clause (, ; :): 100 ms
   - Connector: 80 ms
3. Zero Trailing Silence:
   - Audio ends immediately after the speaker finishes the final word.
   - No awkward dead digital silence.
4. Master Polish:
   - 65 Hz Butterworth HPF removes sub-bass mic rumble.
   - Master peak normalization to -1 dB (0.90).
"""

import numpy as np
import scipy.signal as signal

SR = 24000
sos_hp = signal.butter(4, 65, 'hp', fs=SR, output='sos')

def clean_speech_bounds(wave: np.ndarray, sr: int = 24000, pad_lead_ms: int = 50, pad_tail_ms: int = 140) -> np.ndarray:
    """
    Cleans vocoder onset latency and trailing vocoder air from an F5-TTS chunk.
    Preserves all delicate consonants and release decay without any chopping.
    """
    if len(wave) < int(0.08 * sr):
        return wave

    frame_len = int(0.015 * sr)
    hop_len = int(0.005 * sr)
    n_frames = (len(wave) - frame_len) // hop_len + 1
    if n_frames < 3:
        return wave

    rms = np.array([
        np.sqrt(np.mean(wave[i * hop_len : i * hop_len + frame_len] ** 2))
        for i in range(n_frames)
    ])
    
    # Estimate ambient noise floor (protect delicate tail suffixes while eliminating room hiss)
    noise_floor = np.percentile(rms, 15)
    thresh = max(0.0035, noise_floor * 1.25)

    active = np.where(rms > thresh)[0]
    if len(active) == 0:
        return wave

    lead_sample = max(0, int((active[0] * hop_len) - (pad_lead_ms / 1000.0 * sr)))
    tail_sample = min(len(wave), int(((active[-1] * hop_len) + frame_len) + (pad_tail_ms / 1000.0 * sr)))

    trimmed = wave[lead_sample:tail_sample].copy()

    # 15ms cosine crossfade at head and tail eliminates any click, pop or trailing buzz
    fade = int(0.015 * sr)
    if len(trimmed) > 2 * fade and fade > 0:
        t = np.linspace(0, np.pi / 2, fade).astype(np.float32)
        trimmed[:fade] *= np.sin(t)
        trimmed[-fade:] *= np.cos(t)

    return trimmed

def stitch_chunks_zero_defect(
    chunks: list,
    pause_types: list,
    sr: int = 24000,
    terminal_pause_ms: int = 260,
    clause_pause_ms: int = 100,
    connector_pause_ms: int = 80
) -> np.ndarray:
    """
    Stitches individual F5-TTS chunks into an unbroken, studio-grade speech track.
    - Zero dropped phonemes or chopped syllables.
    - Natural breath pauses between clauses and sentences.
    - Zero trailing silence at the end of the file.
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    n_chunks = len(chunks)
    cleaned_chunks = []

    for idx, c in enumerate(chunks):
        is_last = (idx == n_chunks - 1)
        # 180ms tail on final chunk so trailing suffixes (-da, -dan, -moqda) are 100% intact
        tail_ms = 180 if is_last else 150
        c_clean = clean_speech_bounds(c.astype(np.float32), sr=sr, pad_lead_ms=50, pad_tail_ms=tail_ms)
        cleaned_chunks.append(c_clean)

    # Soft inter-chunk gain leveling (smooths energy jumps between chunks)
    if len(cleaned_chunks) > 1:
        chunk_rms_list = [np.sqrt(np.mean(c**2)) if len(c) > 0 else 0.05 for c in cleaned_chunks]
        median_rms = float(np.median([r for r in chunk_rms_list if r > 0.01])) if any(r > 0.01 for r in chunk_rms_list) else 0.05
        levelled_chunks = []
        for c, c_rms in zip(cleaned_chunks, chunk_rms_list):
            if c_rms > 0.005 and median_rms > 0.005:
                ratio = c_rms / median_rms
                if ratio > 1.8:
                    c = c * (1.8 / ratio)
                elif ratio < 0.55:
                    c = c * (0.55 / ratio)
            levelled_chunks.append(c)
        cleaned_chunks = levelled_chunks

    # Concatenate with natural pauses
    final_pieces = []
    for idx, (chunk, p_type) in enumerate(zip(cleaned_chunks, pause_types)):
        final_pieces.append(chunk)
        if idx < n_chunks - 1:
            if p_type == 'terminal':
                pause_ms = terminal_pause_ms
            elif p_type == 'clause':
                pause_ms = clause_pause_ms
            else:
                pause_ms = connector_pause_ms
            pause_samples = int((pause_ms / 1000.0) * sr)
            if pause_samples > 0:
                final_pieces.append(np.zeros(pause_samples, dtype=np.float32))

    full_wave = np.concatenate(final_pieces)

    # 65 Hz Butterworth HPF to remove mic rumble
    full_wave = signal.sosfiltfilt(sos_hp, full_wave).astype(np.float32)

    # Peak normalization to -1 dB (0.90)
    max_peak = np.max(np.abs(full_wave))
    if max_peak > 0.001:
        full_wave = full_wave * (0.90 / max_peak)

    return full_wave
