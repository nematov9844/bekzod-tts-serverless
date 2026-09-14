#!/usr/bin/env python3
"""
pipeline/phonetic_engine.py
O'zbek Tili Fonetik Vaqt va Davomiylik Dvigateli (Uzbek Phonetic Duration Engine)
- Har bir harf, bo'g'in va morfologik qo'shimchaning tabiiy davomiyligini (ms) hisoblaydi.
- Gap oxiridagi pozitsion cho'zilishni (Pre-pausal lengthening: +40%) hisobga oladi.
- Undoshlar to'qnashuvi va diftonglar uchun mikro-o'tish vaqtini beradi.
- Audio boshlanishidagi tovushlarni (P, Q, U) 100% asraydi va oxiridagi "tq" chertishini yo'qotadi.
"""

import re
import unicodedata
import numpy as np

# Bazaviy fonetik davomiyliklar (millisekundlarda)
PHONEME_DURATIONS_MS = {
    # Cho'ziq va asosiy unlilar
    'a': 115, 'o': 120, 'e': 110,
    # Qisqa unlilar
    'i': 75, 'u': 80,
    # Diftong / tutuq belgili unlilar
    "o'": 135,
    # Sirg'aluvchilar va affrikatlar
    's': 95, 'z': 90, 'sh': 105, 'ch': 100, 'x': 100, 'h': 95, 'f': 90,
    # Portlovchi undoshlar (portlash + oklyuziya)
    'p': 65, 't': 65, 'k': 70, 'q': 75, 'b': 65, 'd': 65, 'g': 70,
    # Sonor va burun tovushlari
    'm': 85, 'n': 85, 'l': 80, 'r': 85, 'y': 80, 'v': 80, 'j': 85,
    # Tutuq belgisi va bo'shliq
    "'": 60,
    ' ': 55
}

# Maxsus ko'p harfli birikmalar (diftong va o'tishlar)
CLUSTER_DURATIONS_MS = {
    "o'y": 180,
    "ay": 150,
    "ey": 140,
    "iy": 130,
    "uy": 140,
    "tashlab qo'y": 950,
    "olib qo'y": 750,
}

# Gap oxirida cho'ziluvchi tipik o'zbekcha suffikslar
TERMINAL_SUFFIXES = [
    "qolibdi", "yetibdi", "bordi", "keldi", "berdi", "tushdi", "qildi",
    "bo'ldi", "chiqdi", "aytgan", "qilingan", "kelmoqda", "bormoqda",
    "qo'y", "debdi"
]

def calculate_phonetic_duration(text: str, is_terminal_sentence: bool = True, speed_factor: float = 1.0) -> float:
    """
    Berilgan o'zbekcha matn uchun ilmiy-fonetik davomiylikni (soniyalarda) hisoblaydi.
    """
    clean_t = text.lower()
    clean_t = unicodedata.normalize("NFC", clean_t)
    clean_t = re.sub(r"[`'ʻʼʽ՚’‘]", "'", clean_t)
    
    # 1. So'zlar va bo'g'inlarni tahlil qilish
    words = clean_t.split()
    if not words:
        return 2.0

    total_ms = 0.0

    for w_idx, word in enumerate(words):
        is_last_word = (w_idx == len(words) - 1)
        word_clean = re.sub(r"[^a-z']", "", word)
        
        # Harflar bo'yicha hisoblash
        i = 0
        w_len = len(word_clean)
        word_ms = 0.0
        
        while i < w_len:
            # 2 ta belgili birikmalar (o', sh, ch)
            if i + 1 < w_len:
                two_char = word_clean[i:i+2]
                if two_char in ["o'", "sh", "ch"]:
                    word_ms += PHONEME_DURATIONS_MS.get(two_char, 100)
                    i += 2
                    continue
                    
            ch = word_clean[i]
            dur = PHONEME_DURATIONS_MS.get(ch, 80)
            
            # Qo'sh undosh (Geminate: mm, tt, bb, dd, ll, ss, kk)
            if i + 1 < w_len and word_clean[i+1] == ch and ch not in "aoeiu":
                dur += 45  # Geminate occlusion hold
                
            # Qo'sh unli (Hiatus: ua, oa, oi, aa, ii, io)
            if i + 1 < w_len and ch in "aoeiu" and word_clean[i+1] in "aoeiu":
                dur += 45  # Hiatus vocal tract reconfiguration
                
            word_ms += dur
            i += 1
            
        # Agar bu so'z gapning oxirgi so'zi bo'lsa (Pre-pausal lengthening: +35% – +45%)
        if is_last_word and is_terminal_sentence:
            # Oxirgi bo'g'in yutilishining oldini olish uchun
            word_ms *= 1.40
            # Agar maxsus suffikslardan biri bo'lsa (-di, -ti, -bdi)
            if any(word_clean.endswith(sfx) for sfx in ["di", "ti", "bdi", "gan", "qoy", "qo'y"]):
                word_ms += 120 # Qo'shimcha rezonans va so'nish buferi
                
        total_ms += word_ms
        
        # So'zlar orasidagi tabiiy masofa
        if not is_last_word:
            total_ms += PHONEME_DURATIONS_MS[' ']
            # Undoshlar to'qnashuvi tekshiruvi (masalan: tashla[b] [q]o'y)
            next_word = words[w_idx + 1] if w_idx + 1 < len(words) else ""
            if word_clean and next_word:
                last_ch = word_clean[-1]
                first_ch = next_word[0]
                if last_ch in "ptkqbdg" and first_ch in "ptkqbdg":
                    total_ms += 45 # Coarticulation mikro-pauzasi
                    
    # Gap oxiridagi nafas chiqarish va vokal so'nish vaqti
    total_ms += 350 if is_terminal_sentence else 180

    # Tezlik koeffitsiyenti (1.0 = normal, >1.0 = tezroq)
    total_sec = (total_ms / 1000.0) / max(0.5, speed_factor)
    return max(1.9, total_sec)

def safe_render_wave(raw_wave: np.ndarray, sr: int = 24000, pad_tail_ms: int = 250, is_terminal: bool = True) -> np.ndarray:
    """
    To'lqinni audio boshini qirqmasdan (P, Q, U larni saqlab)
    va oxiridagi 'tq' chertishini 100% yo'qotib tozalaydi.
    """
    # 1. Boshlanishini QIRQMAYMIZ (s_idx = 0 saqlanadi, bosh harflar 100% omon qoladi)
    # Boshlanishida juda yengil 4ms fade (chertishsiz silliq kirish)
    micro_start_fade = int(0.004 * sr)
    if len(raw_wave) > micro_start_fade:
        raw_wave[:micro_start_fade] *= np.linspace(0.2, 1.0, micro_start_fade)

    # 2. Tugashida: hech qachon ovozni kesib tashlamaymiz!
    # Tabiiy xona sokinligi buferi (250ms) ulaymiz
    pad_samples = int((pad_tail_ms / 1000.0) * sr)
    tail_pad = np.zeros(pad_samples, dtype=np.float32)
    extended = np.concatenate([raw_wave, tail_pad])
    
    # 3. Oxirgi 60ms to'lqinni mutlaq 0.0 ga silliq tushiruvchi kosinus so'ndirgich (Zero-crossing)
    fade_samples = int(0.060 * sr)
    if len(extended) > fade_samples:
        extended[-fade_samples:] *= (1.0 + np.cos(np.linspace(0, np.pi, fade_samples))) / 2.0
        
    return extended
