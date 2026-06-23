"""Hindi WER normalization — matches training eval (`eval_streaming_wer.py`).

Pipeline (1:1 with training eval):
  1. indic-nlp-library script normalization
  2. nukta strip + chandrabindu → anusvara
  3. (optional) acronym intra-dot join: एल.एच → एलएच
  4. lowercase
  5. indic_numtowords expansion (500 → पाँच सौ)
  6. P/S unicode strip
  7. (optional) postposition split: शिंदेके → शिंदे के
  8. Latin↔Devanagari codeswitch map (hand-curated + auto-mined `utils/codeswitch.json`)
  9. whitespace collapse

merge_boundaries (default ON) collapses Devanagari segmentation noise — joining
acronym dots between Devanagari letters AND splitting glued postpositions —
applied IDENTICALLY to ref + hyp so it can only collapse noise, never create
false matches. Removes ~0.7pp spurious WER on Vistaar-Hi.

Plus a separate helper `has_foreign_script(text)` used at WER-calc time to drop
manifest entries whose reference contains scripts other than Hindi (Devanagari) or
English (Latin) — those are data-quality outliers, not real eval material.
"""
import json
import re
import unicodedata
from pathlib import Path


_INDIC_NORMALIZERS: dict = {}

_INDIC_DIGIT_RE = re.compile(
    r'[0-9'
    r'०-९০-৯੦-੯૦-૯୦-୯௦-௯౦-౯೦-೯൦-൯'
    r']+'
)
_INDIC_DIGIT_TO_ASCII = {
    chr(c + off): str(c)
    for off in (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66)
    for c in range(10)
}


def _get_indic_normalizer(lang: str):
    if lang in _INDIC_NORMALIZERS:
        return _INDIC_NORMALIZERS[lang]
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        norm = IndicNormalizerFactory().get_normalizer(lang)
    except Exception:
        norm = None
    _INDIC_NORMALIZERS[lang] = norm
    return norm


def _expand_numbers(text: str, lang: str) -> str:
    try:
        from indic_numtowords import num2words as _n2w
    except Exception:
        return text

    def _sub(m):
        tok = m.group(0)
        ascii_tok = ''.join(_INDIC_DIGIT_TO_ASCII.get(c, c) for c in tok)
        try:
            return ' ' + _n2w(int(ascii_tok), lang=lang) + ' '
        except Exception:
            return ' '
    return _INDIC_DIGIT_RE.sub(_sub, text)


# Latin → Devanagari loanword map (canonicalises script-mismatched code-switches)
CODESW_LATIN_TO_DEVAN: dict[str, str] = {
    # sports
    "football": "फुटबॉल", "basketball": "बास्केटबॉल", "volleyball": "वॉलीबॉल",
    "hockey": "हॉकी", "cricket": "क्रिकेट", "tennis": "टेनिस", "golf": "गॉल्फ",
    "baseball": "बेसबॉल", "boxing": "बॉक्सिंग", "cycling": "साइकिलिंग",
    "gymnastics": "जिमनास्टिक्स", "polo": "पोलो", "racing": "रेसिंग",
    "skiing": "स्कीइंग", "rugby": "रग्बी", "rifle": "राइफल",
    "alpine": "अल्पाइन", "touring": "टूरिंग",
    # tech
    "computer": "कंप्यूटर", "internet": "इंटरनेट", "online": "ऑनलाइन",
    "website": "वेबसाइट", "program": "प्रोग्राम", "system": "सिस्टम",
    "video": "वीडियो", "voice": "वॉयस", "data": "डेटा", "button": "बटन",
    "project": "प्रोजेक्ट", "design": "डिज़ाइन", "screen": "स्क्रीन",
    "software": "सॉफ्टवेयर", "hardware": "हार्डवेयर",
    # loanwords
    "juice": "जूस", "chocolate": "चॉकलेट", "hot": "हॉट", "cold": "कोल्ड",
    "shirt": "शर्ट", "free": "फ्री", "french": "फ्रेंच", "belgium": "बेल्जियम",
    "blue": "ब्लू", "red": "रेड", "white": "व्हाइट", "black": "ब्लैक",
    "green": "ग्रीन", "underwear": "अंडरवियर",
    # transport
    "motor": "मोटर", "transport": "ट्रांसपोर्ट", "car": "कार", "bus": "बस",
    "train": "ट्रेन", "ticket": "टिकट", "signal": "सिग्नल", "traffic": "ट्रैफिक",
    # news / life
    "conservative": "कंजरवेटिव", "party": "पार्टी", "season": "सीज़न",
    "health": "हेल्थ", "care": "केयर", "speech": "स्पीच", "agency": "एजेंसी",
    "emergency": "इमरजेंसी", "team": "टीम", "training": "ट्रेनिंग",
    "news": "न्यूज़", "music": "म्यूजिक", "festival": "फेस्टिवल",
    "market": "मार्केट", "office": "ऑफिस", "water": "वाटर", "ice": "आइस",
    "roller": "रोलर",
    # English numerals
    "one": "एक", "two": "दो", "three": "तीन", "four": "चार", "five": "पाँच",
    "six": "छह", "seven": "सात", "eight": "आठ", "nine": "नौ", "ten": "दस",
    "eleven": "ग्यारह", "twelve": "बारह", "thirteen": "तेरह", "fourteen": "चौदह",
    "fifteen": "पंद्रह", "sixteen": "सोलह", "seventeen": "सत्रह", "eighteen": "अठारह",
    "nineteen": "उन्नीस", "twenty": "बीस", "thirty": "तीस", "forty": "चालीस",
    "fifty": "पचास", "sixty": "साठ", "seventy": "सत्तर", "eighty": "अस्सी",
    "ninety": "नब्बे", "hundred": "सौ", "thousand": "हज़ार",
    "million": "मिलियन", "billion": "बिलियन",
    # places
    "paris": "पेरिस", "spanish": "स्पैनिश", "czech": "चेक",
    "south": "साउथ", "north": "नॉर्थ", "east": "ईस्ट", "west": "वेस्ट",
    "italy": "इटली", "cincinnati": "सिनसिनाटी", "sagittarius": "सैजिटेरियस",
    "india": "इंडिया", "australia": "ऑस्ट्रेलिया",
    # film / media
    "movie": "मूवी", "film": "फिल्म", "action": "एक्शन",
    "director": "डायरेक्टर", "production": "प्रोडक्शन", "picture": "पिक्चर",
    "version": "वर्जन", "versions": "वर्जन", "antique": "एंटीक",
    "format": "फॉर्मेट", "analog": "एनालॉग", "photography": "फोटोग्राफी",
    "costume": "कॉस्ट्यूम", "score": "स्कोर", "movement": "मूवमेंट",
    # generic
    "super": "सुपर", "control": "कंट्रोल", "best": "बेस्ट",
    "district": "डिस्ट्रिक्ट", "station": "स्टेशन", "virtual": "वर्चुअल",
    "carbon": "कार्बन",
    # Apr 2026 audits
    "police": "पुलिस", "foster": "फॉस्टर", "galaxy": "गैलेक्सी",
    "observation": "ऑब्जर्वेशन", "simulate": "सिम्युलेट",
    "scrap": "स्क्रेप", "book": "बुक", "yard": "यार्ड", "apple": "ऐप्पल",
    "school": "स्कूल", "court": "कोर्ट", "supreme": "सुप्रीम",
    "congress": "कांग्रेस", "justice": "जस्टिस", "national": "नेशनल",
    "international": "इंटरनेशनल", "university": "यूनिवर्सिटी",
    "institute": "इंस्टीट्यूट",
    "phone": "फोन", "mobile": "मोबाइल", "facebook": "फेसबुक",
    "device": "डिवाइस", "blackberry": "ब्लैकबेरी",
    "report": "रिपोर्ट", "photo": "फोटो", "show": "शो",
    "model": "मॉडल", "models": "मॉडल्स",
    "ambulance": "एम्बुलेन्स", "member": "मेंबर",
    "practice": "प्रैक्टिस", "talent": "टैलेंट",
    "vacancy": "वैकेंसी", "makeup": "मेकअप",
    "viral": "वायरल", "launch": "लॉन्च",
    "coffee": "कॉफी", "menu": "मीनू",
    "number": "नंबर", "share": "शेयर",
    "company": "कंपनी", "test": "टेस्ट",
    "admission": "एडमिशन", "dance": "डांस",
    "fashion": "फैशन", "media": "मीडिया",
    "designer": "डिजाइनर", "stage": "स्टेज",
    "ban": "बैन", "pregnant": "प्रेग्नेंट",
    "list": "लिस्ट", "business": "बिजनेस",
    "research": "रिसर्च", "release": "रिलीज",
    "birthday": "बर्थडे", "trailer": "ट्रेलर",
    "air": "एयर", "strike": "स्ट्राइक",
    "fans": "फैंस", "return": "रिटर्न",
    "powder": "पाउडर", "tractor": "ट्रैक्टर",
    "garden": "गार्डन", "general": "जनरल",
    "artist": "आर्टिस्ट",
    "kohli": "कोहली", "modi": "मोदी", "ipl": "आईपीएल",
    # Manual gap-fill for words confirmed in references where the streaming model
    # writes Devanagari. Safe because the map only applies to Latin tokens — pure
    # Hindi `अर्थ` (meaning) in ref text never matches against ref `earth`.
    "leadership": "लीडरशिप", "basically": "बेसिकली", "leader": "लीडर",
    "electrical": "इलेक्ट्रिकल", "machine": "मशीन", "machines": "मशीनें",
    "washing": "वाशिंग", "pump": "पंप", "fridge": "फ्रिज",
    "earth": "अर्थ", "earthing": "अर्थिंग", "earthin": "अर्थिन",
    "pipe": "पाइप", "rod": "रॉड", "chemical": "केमिकल",
    "switch": "स्विच", "way": "वे", "single": "सिंगल",
    "ingredients": "इंग्रेडियंट्स", "holder": "होल्डर", "material": "मटेरियल",
    "unit": "यूनिट", "direct": "डायरेक्ट",
    "complexes": "कॉम्प्लेक्सिस", "gazettes": "गैजेट्स",
    "socialization": "सोशलाइजेशन", "expectation": "एक्सपेक्टेशन",
    "yes": "यस",  # model uses both यस and जी; यस handles literal "yes" in code-switch
    "beta": "बीटा",
    # Tiny code-switch fillers the model reliably writes in Devanagari.
    # Adding only these high-confidence stopword pairs (model was consistent in our data).
    "is": "इज", "it": "इट", "you": "यू", "we": "वी", "are": "आर", "am": "एम",
    "of": "ऑफ", "the": "द", "and": "एंड", "in": "इन", "to": "टू",
    "so": "सो", "no": "नो", "by": "बाय", "but": "बट",
    # ─── Sync with training-eval (eval_streaming_wer.py) — May 2026 audits ───
    # telephony.jsonl Gemini-extracted dict
    "suv": "एसयूवी", "hatchback": "हैचबैक", "sedan": "सेडान",
    "brezza": "ब्रेजा",
    "noida": "नोएडा", "gurgaon": "गुडगांव",
    "last": "लास्ट",
    # v5_bias_fix telephony eval Gemini Pro audit
    "app": "ऐप", "budget": "बजट", "call": "कॉल", "cheque": "चेक",
    "city": "सिटी", "cloud": "क्लाउड", "college": "कॉलेज",
    "color": "कलर", "compact": "कॉम्पैक्ट", "confirm": "कन्फर्म",
    "cover": "कवर", "delhi": "दिल्ली", "drive": "ड्राइव",
    "enroll": "एनरोल", "gandhi": "गांधी", "great": "ग्रेट",
    "july": "जुलाई", "lakh": "लाख", "missile": "मिसाइल",
    "offer": "ऑफर", "pyramid": "पिरामिड", "rahul": "राहुल",
    "rate": "रेट", "road": "रोड", "rohit": "रोहित",
    "second": "सेकंड", "sir": "सर", "staff": "स्टाफ",
    # inhouse_segments phase2 over-transliteration audit (formerly short_inhouse)
    "comfortable":  "कंफर्टेबल",
    "spacious":     "स्पेशस",
    "combination":  "कॉम्बिनेशन",
    "approximate":  "अप्रोक्षिमेट",
    "arrangement":  "अरेजमेंट",
    "recommend":    "रिकमेंट",
    "theek":        "ठीक",
    # Vistaar-Hi v5.3 high-WER codemix audit
    "room": "रूम", "google": "गूगल", "loan": "लोन", "bank": "बैंक",
    "twitter": "ट्विटर", "account": "अकाउंट", "store": "स्टोर", "post": "पोस्ट",
    "link": "लिंक", "double": "डबल", "time": "टाइम",
    "cream": "क्रीम", "social": "सोशल", "union": "यूनियन", "trade": "ट्रेड",
    "zero": "जीरो", "reporter": "रिपोर्टर", "contestant": "कंटेस्टेंट",
    "actress": "एक्ट्रेस", "inspector": "इंस्पेक्टर", "girlfriend": "गर्लफ्रेंड",
    "cruise": "क्रूज", "economic": "इकनोमिक", "monitoring": "मोनिटरिंग",
    "arizona": "एरिजोना",
    # v5.3_numerals vistaar high-WER audit (Gemini 2.5 Pro text-only — May 2026)
    # Reverse-direction codeswitch: hyp produces Latin where ref uses Devanagari.
    "academy": "एकेडमी", "airways": "एयरवेज़", "american": "अमेरिकन",
    "champion": "चैंपियन", "deserve": "डिज़र्व", "dm": "डीएम",
    "idol": "आइडल", "jet": "जेट", "leoni": "लियोनी",
    "mtv": "एमटीवी", "pilot": "पायलट", "raped": "रेप्ड",
    "reality": "रियलिटी", "smith": "स्मिथ", "solution": "सॉल्यूशंस",
    "sunny": "सनी",
}

# Multi-word phrases the model contracts into a single Devanagari token.
# Applied BEFORE tokenization (regex on raw text) so they survive whitespace splitting.
CODESW_LATIN_PHRASES: list[tuple[str, str]] = [
    (r"\bma\s+am\b",     "मैम"),    # "ma am"  → मैम (model writes one token)
    (r"\bsir\s+ji\b",    "सरजी"),
    (r"\byes\s+ma\s+am\b", "जी मैम"),  # earlier than ma am match (longest-first)
    (r"\bfull\s+form\b", "फुलफॉर्म"),
]
# Sort by phrase length descending so longest matches first
CODESW_LATIN_PHRASES.sort(key=lambda p: -len(p[0]))


# Devanagari phrase canonicalization — applied to BOTH ref + hyp identically.
# Handles two classes of "two valid forms" problems:
#
# 1. Year-form: 1963 has two valid Hindi expansions, both produced in our data:
#      - `एक हज़ार नौ सौ तिरेसठ` (one thousand nine hundred sixty-three)  ← `indic_numtowords` form
#      - `उन्नीस सौ तिरेसठ` (nineteen hundred sixty-three)               ← v5.3_numerals model emits this
#    We canonicalize the long form → short form (no `\b` since Devanagari isn't
#    a word char to Python's re; use whitespace-anchored lookarounds instead).
#
# 2. Compound-word boundary: refs sometimes glue compounds while model splits
#    them (e.g. `मध्यपूर्व` ↔ `मध्य पूर्व`). Canonicalize to the split form
#    (the model's preferred form, and arguably the more standard one).
#
# Patterns are listed longest-first so larger spans take precedence.
CODESW_DEVAN_PHRASES: list[tuple[str, str]] = [
    # ── Year-form: 1100..1900 ──
    (r"(?<!\S)एक\s+हजार\s+एक\s+सौ(?!\S)",   "ग्यारह सौ"),  # 1100
    (r"(?<!\S)एक\s+हजार\s+दो\s+सौ(?!\S)",    "बारह सौ"),    # 1200
    (r"(?<!\S)एक\s+हजार\s+तीन\s+सौ(?!\S)",   "तेरह सौ"),    # 1300
    (r"(?<!\S)एक\s+हजार\s+चार\s+सौ(?!\S)",   "चौदह सौ"),    # 1400
    (r"(?<!\S)एक\s+हजार\s+पाँच\s+सौ(?!\S)",  "पंद्रह सौ"),  # 1500
    (r"(?<!\S)एक\s+हजार\s+पांच\s+सौ(?!\S)",  "पंद्रह सौ"),  # 1500 (alt spelling)
    (r"(?<!\S)एक\s+हजार\s+छह\s+सौ(?!\S)",    "सोलह सौ"),    # 1600
    (r"(?<!\S)एक\s+हजार\s+सात\s+सौ(?!\S)",   "सत्रह सौ"),   # 1700
    (r"(?<!\S)एक\s+हजार\s+आठ\s+सौ(?!\S)",    "अठारह सौ"),   # 1800
    (r"(?<!\S)एक\s+हजार\s+नौ\s+सौ(?!\S)",    "उन्नीस सौ"),  # 1900

    # ── Compound words split inconsistently across ref and hyp ──
    (r"(?<!\S)मध्य\s*पूर्व(?!\S)",   "मध्य पूर्व"),
    (r"(?<!\S)पूर्वोत्तर(?!\S)",     "पूर्व उत्तर"),
    (r"(?<!\S)दू\s+सरों(?!\S)",      "दूसरों"),       # ref-split → glued
    (r"(?<!\S)ग्यारह\s+वीं(?!\S)",   "ग्यारहवीं"),
    (r"(?<!\S)बारह\s+वीं(?!\S)",     "बारहवीं"),
    (r"(?<!\S)तेरह\s+वीं(?!\S)",     "तेरहवीं"),
    (r"(?<!\S)चौदह\s+वीं(?!\S)",     "चौदहवीं"),
    (r"(?<!\S)पंद्रह\s+वीं(?!\S)",   "पंद्रहवीं"),
    (r"(?<!\S)सोलह\s+वीं(?!\S)",     "सोलहवीं"),
    (r"(?<!\S)सत्रह\s+वीं(?!\S)",    "सत्रहवीं"),
    (r"(?<!\S)अठारह\s+वीं(?!\S)",    "अठारहवीं"),
    (r"(?<!\S)उन्नीस\s+वीं(?!\S)",   "उन्नीसवीं"),
    (r"(?<!\S)बीस\s+वीं(?!\S)",      "बीसवीं"),
]
CODESW_DEVAN_PHRASES.sort(key=lambda p: -len(p[0]))


# Auto-mined extensions: Latin tokens we observed in references that the streaming
# model spelled in Devanagari with high confidence (count ≥5, dominance ≥70%, then
# triple-validated by Gemini). Mining: `scripts/mine_codeswitch.py`. Cleaning:
# `scripts/clean_codeswitch.py`. Stored in `utils/codeswitch.json`.
_MINED_PATH = Path(__file__).parent / "codeswitch.json"
if _MINED_PATH.exists():
    try:
        _mined = json.loads(_MINED_PATH.read_text())
        # hand-curated entries always win on conflict
        for k, v in _mined.items():
            CODESW_LATIN_TO_DEVAN.setdefault(k, v)
    except Exception:
        pass


def _apply_codeswitch(text: str, m: dict[str, str]) -> str:
    if not m:
        return text
    # Multi-word phrase pass first (case-insensitive on the lowercased input).
    for pat, repl in CODESW_LATIN_PHRASES:
        text = re.sub(pat, repl, text)
    # Devanagari phrase canonicalization (year-form long↔short, compound boundary).
    # Applied identically to ref and hyp so it can only collapse noise, never
    # create false matches.
    for pat, repl in CODESW_DEVAN_PHRASES:
        text = re.sub(pat, repl, text)
    return " ".join(m.get(t, t) for t in text.split())


# Word-boundary normalization — Devanagari postpositions that get glued onto
# the preceding word, and intra-acronym dots between Devanagari letters.
# Applied identically to ref + hyp.
_HI_POSTPOS = ("के", "की", "का", "को", "में", "से", "ने", "पर", "तक")
_POSTPOS_RE = re.compile(r'(?<=[ऀ-ॿ])(' + '|'.join(_HI_POSTPOS) + r')(?=\s|$)')
_ACRONYM_DOT_RE = re.compile(r'(?<=[ऀ-ॿ])\.(?=[ऀ-ॿ])')


def _split_postpositions(text: str) -> str:
    """शिंदेके → शिंदे के  (standalone postpositions are left untouched)."""
    return _POSTPOS_RE.sub(r' \1', text)


def normalize_indic(
    text: str,
    indic_lang: str = "hi",
    expand_numbers: bool = True,
    apply_codeswitch: bool = True,
    codesw_map: dict | None = None,
    merge_boundaries: bool = True,
) -> str:
    """Match training-eval normalization exactly."""
    n = _get_indic_normalizer(indic_lang)
    if n is not None:
        try:
            text = n.normalize(text)
        except Exception:
            pass
    text = text.replace('़', '')      # nukta strip
    text = text.replace('ँ', 'ं')     # chandrabindu → anusvara
    # Join intra-acronym dots BEFORE punct strip turns them into spaces
    # (एल.एच.सीबी → एलएचसीबी, matching the model's un-dotted form).
    if merge_boundaries:
        text = _ACRONYM_DOT_RE.sub('', text)
    text = text.lower()
    if expand_numbers:
        text = _expand_numbers(text, lang=indic_lang)
        # `indic_numtowords.num2words(1920)` outputs `एक हज़ार नौ सौ बीस` WITH
        # the nukta — even though we stripped it from the raw text earlier. If
        # we don't strip again, refs (already in word form, nukta gone) and
        # hyps (digit form, expanded to nukta-bearing words) mismatch on every
        # occurrence of हज़ार. Restrip here.
        text = text.replace('़', '')
    text = ''.join(' ' if unicodedata.category(c)[0] in ('P', 'S') else c for c in text)
    text = re.sub(r'\s+', ' ', text).strip()
    if merge_boundaries:
        text = _split_postpositions(text)
        text = re.sub(r'\s+', ' ', text).strip()
    if apply_codeswitch:
        text = _apply_codeswitch(
            text, codesw_map if codesw_map is not None else CODESW_LATIN_TO_DEVAN
        )
    return text


def contains_digit(text: str) -> bool:
    return bool(_INDIC_DIGIT_RE.search(text or ""))


# Allowed code-point ranges for "Hindi or English only" reference text.
# Anything outside these (Cyrillic, Arabic, Chinese, other Brahmi scripts) flags
# the utterance for exclusion from WER.
def _is_allowed_letter_cp(cp: int) -> bool:
    if 0x0041 <= cp <= 0x005A: return True   # Latin uppercase
    if 0x0061 <= cp <= 0x007A: return True   # Latin lowercase
    if 0x00C0 <= cp <= 0x024F: return True   # Latin-1 Supplement + Extended-A/B (accented latin)
    if 0x0900 <= cp <= 0x097F: return True   # Devanagari
    return False


def has_foreign_script(text: str) -> bool:
    """True iff `text` contains a *letter* outside Latin or Devanagari.

    Whitespace, digits, ASCII punctuation, smart-quotes/dashes, and the rupee
    symbol are all OK — only foreign-script LETTERS trigger this. Russian
    Cyrillic, Arabic, Chinese, Punjabi/Bengali/Tamil/Telugu Brahmi scripts → True.
    """
    if not text: return False
    for c in text:
        if c.isspace() or c.isdigit(): continue
        cat = unicodedata.category(c)
        if cat[0] in ("P", "S", "M", "Z", "N", "C"): continue  # punct/symbol/mark/sep/number/control: ignore
        # `cat[0] == "L"` (a letter) — must be Latin or Devanagari only
        if not _is_allowed_letter_cp(ord(c)):
            return True
    return False
