#!/usr/bin/env python3
"""
pipeline/cyrillic_to_latin.py
==============================
O'zbekcha kirill matnini rasmiy lotin alifbosiga o'giradi (davlat standarti asosida).
"""

import re

BASE_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sh", "ъ": "'", "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o'", "ғ": "g'", "қ": "q", "ҳ": "h",
}
VOWELS_CYR = set("аеёиоуыэюя")


def _translit_word(word: str) -> str:
    out = []
    for idx, ch in enumerate(word):
        lower = ch.lower()
        is_upper = ch.isupper()
        if lower == "е":
            prev = word[idx - 1].lower() if idx > 0 else None
            if idx == 0 or (prev and (prev in VOWELS_CYR or prev in "ъь")):
                rep = "ye"
            else:
                rep = "e"
        else:
            rep = BASE_MAP.get(lower, lower)
        if is_upper and rep:
            rep = rep[0].upper() + rep[1:]
        out.append(rep)
    return "".join(out)


def cyrillic_to_latin(text: str) -> str:
    return re.sub(r"[А-Яа-яЁёЎўҒғҚқҲҳ]+", lambda m: _translit_word(m.group(0)), text)


if __name__ == "__main__":
    sample = "Ҳурматли тингловчилар, божхона тўловларини ўз вақтида тўлаш керак."
    print(cyrillic_to_latin(sample))
