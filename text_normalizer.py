#!/usr/bin/env python3
"""
Uzbek Text Normalizer for TTS Pipeline
Qo'llab-quvvatlaydi:
- Raqamlarni so'zga aylantirish (butun, o'nli, tartib raqamlar, yillar, soatlar, foizlar)
- Valyutalar (so'm, dollar, yevro, rubl)
- Tutuq belgilari va maxsus harflar (o', g', sh, ch, ng standartlashtirish)
- Katta harflar va maxsus belgilarni tozalash
"""

import re
import unicodedata

ONES = {
    0: "nol", 1: "bir", 2: "ikki", 3: "uch", 4: "to'rt",
    5: "besh", 6: "olti", 7: "yetti", 8: "sakkiz", 9: "to'qqiz"
}
TENS = {
    10: "o'n", 20: "yigirma", 30: "o'ttiz", 40: "qirq",
    50: "ellik", 60: "oltmish", 70: "yetmish", 80: "sakson", 90: "to'qson"
}
SCALES = [
    (10**12, "trillion"),
    (10**9, "milliard"),
    (10**6, "million"),
    (10**3, "ming"),
    (10**2, "yuz")
]

def integer_to_uzbek(num: int, is_root: bool = True) -> str:
    """Butun sonni o'zbekcha so'zga aylantiradi."""
    if num == 0:
        return ONES[0]
    if num < 0:
        return "minus " + integer_to_uzbek(abs(num), is_root=False)
    
    parts = []
    
    for scale, name in SCALES:
        if num >= scale:
            count = num // scale
            num %= scale
            if count == 1:
                if num == 0 and len(parts) == 0:
                    parts.append(name)
                else:
                    parts.append("bir " + name)
            else:
                parts.append(integer_to_uzbek(count, is_root=False) + " " + name)
                
    if num >= 10:
        ten = (num // 10) * 10
        parts.append(TENS[ten])
        num %= 10
        
    if num > 0:
        parts.append(ONES[num])
        
    return " ".join(parts)

def normalize_numbers(text: str) -> str:
    """Matndagi barcha raqamlarni so'zga almashtiradi."""
    # Guruhlangan raqamlarni birlashtirish: 1 250 000 yoki 1,250,000 -> 1250000
    text = re.sub(r'(?<=\d)[\s,._](?=\d{3}\b)', '', text)

    # Foizlar va o'nli kasrlar: 8.5% yoki 8,5% -> sakkiz butun besh foiz
    text = re.sub(r'(\d+)[.,](\d+)\s*%', lambda m: f"{integer_to_uzbek(int(m.group(1)))} butun {integer_to_uzbek(int(m.group(2)))} foiz", text)
    # Foizlar: 50% -> ellik foiz
    text = re.sub(r'(\d+)\s*%', lambda m: integer_to_uzbek(int(m.group(1))) + " foiz", text)
    
    # O'nli kasrlar: 7.8 yoki 0.75 -> yetti butun sakkiz
    text = re.sub(r'(\d+)[.,](\d+)', lambda m: f"{integer_to_uzbek(int(m.group(1)))} butun {integer_to_uzbek(int(m.group(2)))}", text)
    
    # Pul birliklari
    text = re.sub(r'\$(\d+)', lambda m: integer_to_uzbek(int(m.group(1))) + " dollar", text)
    text = re.sub(r'(\d+)\s*so\'?m', lambda m: integer_to_uzbek(int(m.group(1))) + " so'm", text)
    text = re.sub(r'(\d+)\s*rubl', lambda m: integer_to_uzbek(int(m.group(1))) + " rubl", text)
    text = re.sub(r'(\d+)\s*yevro', lambda m: integer_to_uzbek(int(m.group(1))) + " yevro", text)
    
    # Vaqt: 14:30 -> o'n to'rt o'ttiz
    text = re.sub(r'(\d{1,2}):(\d{2})', lambda m: f"{integer_to_uzbek(int(m.group(1)))} {integer_to_uzbek(int(m.group(2)))}", text)
    
    # Tartib raqamlar: 1-chi, 2-chi, 5-avgust -> birinchi, ikkinchi, beshinchi
    ORDINAL_SUFFIXES = {
        "bir": "birinchi", "ikki": "ikkinchi", "uch": "uchinchi", "to'rt": "to'rtinchi",
        "besh": "beshinchi", "olti": "oltinchi", "yetti": "yettinchi", "sakkiz": "sakkizinchi",
        "to'qqiz": "to'qqizinchi", "o'n": "o'ninchi", "yigirma": "yigirmanchi", "o'ttiz": "o'ttizinchi",
        "qirq": "qirqinchi", "ellik": "elliginchi", "oltmish": "oltmishinchi", "yetmish": "yetmishinchi",
        "sakson": "saksoninchi", "to'qson": "to'qsoninchi", "yuz": "yuzinchi", "ming": "minginchi"
    }
    
    def make_ordinal(m):
        num_str = integer_to_uzbek(int(m.group(1)))
        words = num_str.split()
        last_word = words[-1]
        if last_word in ORDINAL_SUFFIXES:
            words[-1] = ORDINAL_SUFFIXES[last_word]
        elif last_word.endswith(('a', 'i', 'e', 'o', 'u')):
            words[-1] = last_word + "nchi"
        else:
            words[-1] = last_word + "inchi"
        return " ".join(words)

    # Yillar: 2026-yil / 2026 yil -> ikki ming yigirma oltinchi yil
    text = re.sub(r'(\d+)-(?:yil|yilda|yilgi|yildan)', lambda m: make_ordinal(m) + " " + m.group(0).split('-')[-1], text)
    text = re.sub(r'(\d+)\s+(yil|yilda|yilgi|yildan)\b', lambda m: make_ordinal(m) + " " + m.group(2), text)

    # Fonetik tuzatish: o'zbek adabiy va jonli talaffuzida sentiyabr / oktabr shaklida to'liq aytiladi
    months = r'(yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sentabr|sentiyabr|oktabr|oktyabr|noyabr|dekabr)'
    MONTH_PHONETIC = {'sentabr': 'sentiyabr', 'sentyabr': 'sentiyabr', 'oktabr': 'oktabr', 'oktyabr': 'oktabr'}
    def _month_rep(m):
        raw_m = m.group(2).lower()
        return make_ordinal(m) + " " + MONTH_PHONETIC.get(raw_m, raw_m)

    text = re.sub(r'(\d+)-' + months, _month_rep, text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s+' + months, _month_rep, text, flags=re.IGNORECASE)

    # Oddiy raqamlar: 123 -> bir yuz yigirma uch
    text = re.sub(r'\b\d+\b', lambda m: f" {integer_to_uzbek(int(m.group(0)))} ", text)
    # Satrlar ichidagi probellarni tozalash, lekin yangi qatorlarni (\n) saqlash
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    text = '\n'.join(l for l in lines if l).strip()
    
    return text

def normalize_uzbek_characters(text: str) -> str:
    """O'zbek harflari (o', g', tutuq belgilari)ni standartlashtiradi."""
    text = unicodedata.normalize('NFC', text)
    
    # Har xil apostroflarni standart ' (ASCII 39) ga keltirish
    text = re.sub(r"[`'ʻʼʽ՚’‘]", "'", text)
    
    # Whisper chiqaradigan turkiy/ozarbayjon harflarini o'zbekchaga o'girish
    TURKIC_MAP = {
        'ə': 'a', 'Ə': 'A',
        'ş': 'sh', 'Ş': 'Sh',
        'ç': 'ch', 'Ç': 'Ch',
        'ğ': "g'", 'Ğ': "G'",
        'ı': 'i', 'I': 'I',
        'ö': "o'", 'Ö': "O'",
        'ü': 'u', 'Ü': 'U',
    }
    for k, v in TURKIC_MAP.items():
        text = text.replace(k, v)
    
    # Accent and stress vowel mapping (á, ó, é, í, ú -> a, o, e, i, u with stress metadata)
    ACCENTED_VOWEL_MAP = {
        'á': 'a', 'Á': 'A',
        'ó': 'o', 'Ó': 'O',
        'é': 'e', 'É': 'E',
        'í': 'i', 'Í': 'I',
        'ú': 'u', 'Ú': 'U',
        'à': 'a', 'À': 'A',
        'ò': 'o', 'Ò': 'O',
        'è': 'e', 'È': 'E',
        'ì': 'i', 'Ì': 'I',
        'ù': 'u', 'Ù': 'U',
        'â': 'a', 'Â': 'A',
        'ô': 'o', 'Ô': 'O',
        'ê': 'e', 'Ê': 'E',
        'î': 'i', 'Î': 'I',
        'û': 'u', 'Û': 'U',
    }
    for k, v in ACCENTED_VOWEL_MAP.items():
        text = text.replace(k, v)

    # O' va G' harflarini standartlashtirish
    text = re.sub(r"o['']", "o'", text, flags=re.IGNORECASE)
    text = re.sub(r"g['']", "g'", text, flags=re.IGNORECASE)
    
    # Ko'p uchraydigan qisqartmalar
    ABBREVIATIONS = {
        r"\bmas:\b": "masalan",
        r"\bva h\.k\b": "va hokazo",
        r"\bva b\.\b": "va boshqalar",
        r"\bprof\.\b": "professor",
        r"\bdr\.\b": "doktor",
        r"\bt\.y\.\b": "tug'ilgan yili",
        r"\bk\.m\.\b": "kilometr",
        r"\bsm\b": "santimetr",
        r"\bkg\b": "kilogramm",
        r"\bmln\b": "million",
        r"\bmlrd\b": "milliard",
        r"\bAQSH\b": "amerika qo'shma shtatlari",
        r"\bO'zR\b": "o'zbekiston respublikasi",
        r"\bAI\b": "ey ay",
        r"\bIT\b": "ay ti",
        r"\bTTS\b": "te te es",
        r"\baudio": "avdio",
        r"\bsentabr\b": "sentiyabr",
        r"\bsentyabr\b": "sentiyabr",
    }
    
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(abbr, full, text, flags=re.IGNORECASE)
        
    return text

def clean_punctuation_and_whitespace(text: str) -> str:
    """Ortiqcha simvollar, probellar va tinish belgilarini tozalash (yangi qatorlar saqlanadi)."""
    # HACK FIX: Whisper timestamp formatida `▁` aslida bo'sh joy (probel) bo'lib xizmat qiladi
    if '▁' in text:
        text = text.replace(' ', '')
        text = text.replace('▁', ' ')
    
    # Nuqtali vergul (;) va ikki nuqta (:) ni clause chegarasi sifatida saqlash
    text = text.replace(';', ',')
    text = re.sub(r'[\t\r]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    # Defis va chiziqchalarni probelga almashtirish
    text = re.sub(r'[-–—_]+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\!\?\:\'\"]', ' ', text)
    # Satrlar ichidagi probellarni tozalash, lekin yangi qatorlarni saqlash
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    text = '\n'.join(l for l in lines if l).strip()
    return text

_FONETIK_PAIRS_CACHE = None

def get_fonetik_hodisalar_pairs():
    """uzbek_tts_fonetik_hodisalar_1200.md faylidan xavfsiz lotin fonetik qoidalarini keshlaydi."""
    global _FONETIK_PAIRS_CACHE
    if _FONETIK_PAIRS_CACHE is not None:
        return _FONETIK_PAIRS_CACHE
    pairs = []
    try:
        from pathlib import Path
        md_file = Path(__file__).resolve().parent.parent / "uzbek_tts_fonetik_hodisalar_1200.md"
        if not md_file.exists():
            md_file = Path("uzbek_tts_fonetik_hodisalar_1200.md")
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("|") and not line.startswith("| #") and not line.startswith("|---"):
                        parts = [p.strip().strip("`") for p in line.split("|")[1:-1]]
                        if len(parts) >= 3:
                            adabiy, hint = parts[1], parts[2]
                            # Faqat o'zbek lotin alifbosidagi xavfsiz so'zlashuv o'zgarishlarini olish (IPA emas)
                            if any(c in hint for c in "ɪəşçğöü"):
                                continue
                            clean_a = re.sub(r"[^a-zA-Z'\s]", "", adabiy).lower()
                            clean_h = re.sub(r"[^a-zA-Z'\s]", "", hint).lower()
                            if clean_a and clean_h and clean_a != clean_h:
                                pat = r"\b" + re.escape(clean_a) + r"\b"
                                pairs.append((re.compile(pat, re.IGNORECASE), clean_h))
    except Exception:
        pass
    _FONETIK_PAIRS_CACHE = pairs
    return pairs

def apply_uzbek_orthoepy_and_dialect(text: str, style: str = "adabiy") -> str:
    """
    O'zbek tili Orfoepiya va Lahja Fonetik Qoidalari:
    - style="adabiy": Standart adabiy talaffuz (hozirgi sifatli, to'liq adabiy talaffuz).
    - style="fonetik" / "jonli": uzbek_tts_fonetik_hodisalar_1200.md asosidagi jonli qisqarishlar (borgandi, bolsayam, blan).
    - style="toshkent": Toshkent lahjasi (man, san, Toshkenda, qilyapman).
    """
    if style in ("fonetik", "jonli"):
        # 1. 1200 ta fonetik hodisalar jadvali qoidalari (borsa ham -> borsayam, degan edi -> degandi)
        for pat, repl in get_fonetik_hodisalar_pairs():
            text = pat.sub(repl, text)

        # 2. Qo'shimcha morfemik qisqarishlar
        text = re.sub(r'(\w+)sa\s+ham\b', r'\1sayam', text, flags=re.IGNORECASE)
        text = re.sub(r'(\w+)(?:gan|kan|qan)\s+edi\b', r'\1gandi', text, flags=re.IGNORECASE)
        text = re.sub(r'\bbilan\b', 'blan', text, flags=re.IGNORECASE)

    elif style == "toshkent":
        # Olmoshlar
        text = re.sub(r'\bmening\b', 'maning', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmenga\b', 'manga', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmenda\b', 'manda', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmendan\b', 'mandan', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmen\b', 'man', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\bsening\b', 'saning', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsenga\b', 'sanga', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsenda\b', 'sanda', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsendan\b', 'sandan', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsen\b', 'san', text, flags=re.IGNORECASE)
        
        # Shahar va joy nomlari assimilyatsiyasi: Toshkentda -> Toshkenda
        text = re.sub(r'\btoshkentda\b', 'toshkenda', text, flags=re.IGNORECASE)
        text = re.sub(r'\btoshkentga\b', 'toshkenga', text, flags=re.IGNORECASE)
        text = re.sub(r'\btoshkentdan\b', 'toshkendan', text, flags=re.IGNORECASE)
        text = re.sub(r'\btoshkentlik\b', 'toshkenlik', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsamarqandda\b', 'samarqanda', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsamarqandga\b', 'samarqanga', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsamarqanddan\b', 'samarqandan', text, flags=re.IGNORECASE)
        
        # Fe'l shakllari va qisqarishlar
        text = re.sub(r'\byashayman\b', 'yashiman', text, flags=re.IGNORECASE)
        text = re.sub(r'\bkelayapman\b', 'kelyapman', text, flags=re.IGNORECASE)
        text = re.sub(r'\bborayapman\b', 'boryapman', text, flags=re.IGNORECASE)
        text = re.sub(r'\bqilayapman\b', 'qilyapman', text, flags=re.IGNORECASE)
        text = re.sub(r'\baytayapman\b', 'aytyapman', text, flags=re.IGNORECASE)
        text = re.sub(r'\bo\'ylayapman\b', 'o\'ylyapman', text, flags=re.IGNORECASE)
        text = re.sub(r'\bnima\s+qilayapsiz\b', 'nima qilyapsiz', text, flags=re.IGNORECASE)
        
        # So'zlashuv qisqarishlari
        text = re.sub(r'\bhaqiqatdan\b', 'haqiqatan', text, flags=re.IGNORECASE)
        text = re.sub(r'\bkerak\b', 'kerek', text, flags=re.IGNORECASE)
        text = re.sub(r'\bemas\b', 'mas', text, flags=re.IGNORECASE)
        
    # Unlilar assimilyatsiyasi: havo so'zi tabiiy ochiq 'o' bilan aytiladi
    # text = re.sub(r'\bhavo\b(?=\s+[aeiou])', "havo'", text, flags=re.IGNORECASE)

    # 1. Chet tili va o'zlashma so'zlarda io/ia/ie diftonglari (audyo, vidyo, radyo, materyal)
    text = re.sub(r'\baudio', 'audyo', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvideo', 'vidyo', text, flags=re.IGNORECASE)
    text = re.sub(r'\bradio', 'radyo', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstudio', 'studyo', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstadion', 'stadyon', text, flags=re.IGNORECASE)
    text = re.sub(r'\bchempion', 'chempyon', text, flags=re.IGNORECASE)
    text = re.sub(r'\bregion', 'regyon', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmillion', 'milyon', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmilliard', 'milyard', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbilliard', 'bilyard', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmaterial', 'materyal', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvariant', 'varyant', text, flags=re.IGNORECASE)
    text = re.sub(r'\baviatsiy', 'avyatsiy', text, flags=re.IGNORECASE)
    text = re.sub(r'\baviasiy', 'avyatsiy', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpianino', 'pyanino', text, flags=re.IGNORECASE)

    # 2. Hiatus va yonma-yon unlilar mustahkamlanishi (oiila, doiira, shoiir)
    text = re.sub(r'\boila', 'oiila', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdoira', 'doiira', text, flags=re.IGNORECASE)
    text = re.sub(r'\bshoir', 'shoiir', text, flags=re.IGNORECASE)
    text = re.sub(r'\brais\b', 'raiis', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfoiz\b', 'foiiz', text, flags=re.IGNORECASE)

    # 3. Raqamlar va yuzliklar ritmik birikishi (to'rtyuz, beshyuz, uchyuz)
    text = re.sub(r'\b(bir|ikki|uch|to\'rt|besh|olti|yetti|sakkiz|to\'qqiz)\s+yuz\b', r'\1yuz', text, flags=re.IGNORECASE)

    # 4. Kontakt va progressiv/regressiv fonetik assimilatsiya (shamba, tussiz, kitopka)
    text = re.sub(r'\bshanba\b', 'shamba', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdushanba\b', 'dushamba', text, flags=re.IGNORECASE)
    text = re.sub(r'\bseshanba\b', 'seshamba', text, flags=re.IGNORECASE)
    text = re.sub(r'\bchorshanba\b', 'chorshamba', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpayshanba\b', 'payshamba', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmanba\b', 'mamba', text, flags=re.IGNORECASE)
    text = re.sub(r'\byonbosh\b', 'yombosh', text, flags=re.IGNORECASE)
    text = re.sub(r'\btuzsiz', 'tussiz', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(ayt|ot|ket|tut|kut|yot)gan\b', r'\1kan', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(kitob|maktab|hisob|sabab|asbob)ga\b', r'\1ka', text, flags=re.IGNORECASE)
    text = re.sub(r'\buchta\b', 'ushta', text, flags=re.IGNORECASE)
    text = re.sub(r'\byuzta\b', 'yusta', text, flags=re.IGNORECASE)

    # 5. Texnik va ilmiy uzun qo'shma so'zlarni morfemalarga ajratish (urg'u va talaffuz ravshanligi)
    text = re.sub(r'\bneyrotarmoq', 'neyro tarmoq', text, flags=re.IGNORECASE)
    text = re.sub(r'\bneyrobiolog', 'neyro biolog', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnanotexnolog', 'nano texnolog', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbiotibbiyot', 'bio tibbiyot', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbiotexnolog', 'bio texnolog', text, flags=re.IGNORECASE)
    text = re.sub(r'\bkiberxavfsiz', 'kiber xavfsiz', text, flags=re.IGNORECASE)
    text = re.sub(r'\belektromobil', 'elektro mobil', text, flags=re.IGNORECASE)
    text = re.sub(r'\binfratuzilma', 'infra tuzilma', text, flags=re.IGNORECASE)
    text = re.sub(r'\btelekommunikatsiya', 'tele kommunikatsiya', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgidroelektr', 'gidro elektr', text, flags=re.IGNORECASE)
    text = re.sub(r'\bavtotransport', 'avto transport', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmikroiqtisod', 'mikro iqtisod', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmakroiqtisod', 'makro iqtisod', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsuperkompyuter', 'super kompyuter', text, flags=re.IGNORECASE)
    text = re.sub(r'\bekotizim', 'eko tizim', text, flags=re.IGNORECASE)
    text = re.sub(r'\bumumta\'lim', 'umum ta\'lim', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsun\'iyintellekt', 'sun\'iy intellekt', text, flags=re.IGNORECASE)
    text = re.sub(r'\baudiokitob', 'audyo kitob', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvideodars', 'vidyo dars', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvideorolik', 'vidyo rolik', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvideokonferensiya', 'vidyo konferensiya', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvebsayt', 'veb sayt', text, flags=re.IGNORECASE)


    return text

def normalize_uzbek_text(text: str, style: str = "adabiy") -> str:
    """TTS uchun matnni to'liq normalizatsiya qiladi."""
    try:
        try:
            from restore_uzbek_orthography import restore_orthography
        except ImportError:
            from pipeline.restore_uzbek_orthography import restore_orthography
        text = restore_orthography(text)
    except Exception:
        pass
    text = normalize_uzbek_characters(text)
    text = normalize_numbers(text)
    text = clean_punctuation_and_whitespace(text)
    text = apply_uzbek_orthoepy_and_dialect(text, style=style)
    return text

if __name__ == "__main__":
    test_text = "Men Toshkentda yashayman. Mening ismim Bekzod."
    print("Asl matn:      ", test_text)
    print("Adabiy shakl:  ", normalize_uzbek_text(test_text, style="adabiy"))
    print("Toshkent shakli:", normalize_uzbek_text(test_text, style="toshkent"))
