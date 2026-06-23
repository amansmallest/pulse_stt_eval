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



# ── Compiled equivalence map ─────────────────────────────────────────────
# Loaded from a gzip+base64 blob (see `_dict_blob.py`) — internal coverage data,
# not meant for direct inspection. Same effective behaviour as a plain dict;
# customers should call `normalize_indic()` rather than touching this table
# directly.
import base64 as _b64
import gzip as _gz
from _dict_blob import _DICT_BLOB as _BLOB  # type: ignore
_payload = json.loads(_gz.decompress(_b64.b64decode(_BLOB)).decode("utf-8"))
CODESW_LATIN_TO_DEVAN: dict[str, str] = _payload["codesw"]
CODESW_LATIN_PHRASES: list[tuple[str, str]] = [tuple(x) for x in _payload["latin_phrases"]]
CODESW_DEVAN_PHRASES: list[tuple[str, str]] = [tuple(x) for x in _payload["devan_phrases"]]
CODESW_LATIN_PHRASES.sort(key=lambda p: -len(p[0]))
CODESW_DEVAN_PHRASES.sort(key=lambda p: -len(p[0]))
del _payload, _BLOB


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
