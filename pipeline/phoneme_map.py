"""
Two-stage phoneme mapping:
  Stage 1 — Mizo G2P: Latin orthography → IPA string
  Stage 2 — Rengmitca mapping: IPA phones → Rengmitca inventory

Phones not present in the Rengmitca inventory are substituted with the
closest near-match where possible, or flagged as uncertain ('?') otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Stage 1 — Mizo G2P
# Rules are applied left-to-right, longest match first.
# ---------------------------------------------------------------------------

# Each entry: (grapheme_pattern, ipa_string)
# Order matters: longer/more specific patterns must precede shorter ones.
_MIZO_G2P_RULES: list[tuple[str, str]] = [
    # --- Digraph consonants (must precede single-letter matches) ---
    ("ng",  "ŋ"),
    ("ph",  "pʰ"),
    ("kh",  "kʰ"),
    ("th",  "tʰ"),
    ("ch",  "tʃ"),
    ("hl",  "ɬ"),      # voiceless lateral fricative
    ("hn",  "n"),      # aspirated nasal → plain n
    ("hm",  "m"),      # aspirated nasal → plain m
    ("hr",  "r"),
    ("hw",  "w"),
    ("tl",  "tˡ"),     # lateral affricate
    # --- Single consonants ---
    ("p",   "p"),
    ("b",   "b"),
    ("t",   "t"),
    ("d",   "d"),
    ("k",   "k"),
    ("g",   "ɡ"),
    ("c",   "tʃ"),
    ("j",   "dʒ"),
    ("m",   "m"),
    ("n",   "n"),
    ("f",   "f"),
    ("v",   "v"),
    ("s",   "s"),
    ("z",   "z"),
    ("h",   "h"),
    ("l",   "l"),
    ("r",   "r"),
    ("w",   "w"),
    ("y",   "j"),
    # --- Vowels ---
    ("â",   "ɔ"),     # Mizo 'aw-like' vowel written with circumflex
    ("a",   "a"),
    ("e",   "e"),
    ("i",   "i"),
    ("o",   "o"),
    ("u",   "u"),
]

# Build a regex that matches any grapheme in priority order (longest first)
_G2P_PATTERN = re.compile(
    "|".join(re.escape(g) for g, _ in _MIZO_G2P_RULES),
    re.IGNORECASE,
)
_G2P_MAP = {g: ipa for g, ipa in _MIZO_G2P_RULES}


def mizo_to_ipa(text: str) -> list[str]:
    """
    Convert a Mizo word (Latin orthography) to a list of IPA phones.
    Unrecognised characters are passed through as-is.
    """
    phones: list[str] = []
    pos = 0
    lower = text.lower()
    while pos < len(lower):
        match = _G2P_PATTERN.match(lower, pos)
        if match:
            grapheme = match.group(0).lower()
            # Find the matching rule (IGNORECASE, so match lowercased)
            for g, ipa in _MIZO_G2P_RULES:
                if g == grapheme:
                    phones.append(ipa)
                    break
            pos = match.end()
        else:
            # Unknown character — skip whitespace, pass others through
            ch = lower[pos]
            if not ch.isspace():
                phones.append(ch)
            pos += 1
    return phones


# ---------------------------------------------------------------------------
# Stage 2 — Rengmitca inventory mapping
# ---------------------------------------------------------------------------

# Phones that exist verbatim in the Rengmitca inventory are kept as-is.
# Others are mapped to the nearest member of the inventory.
_NEAR_MATCH: dict[str, str] = {
    # Voiced stops → devoiced (Rengmitca lacks voiced obstruents)
    "b":   "p",
    "d":   "t",
    "ɡ":   "k",
    # Aspirates not in Rengmitca inventory
    "kʰ":  "k",
    # Affricates
    "tʃ":  "c",    # palatal affricate
    "dʒ":  "ɟ",
    "tˡ":  "t",    # lateral affricate → plain stop
    # Fricatives not in inventory
    "f":   "s",    # labiodental → alveolar (both voiceless fricatives)
    "v":   "w",    # voiced labiodental → labial-velar
    "z":   "s",    # voiced → devoiced alveolar
    "ɬ":   "s",    # voiceless lateral fricative → s
    # Rhotic
    "r":   "ɹ",
    # Vowel near-matches
    "ɛ":   "e",
    "ə":   "ɘ",
    "ɜ":   "ɘ",
    "ø":   "e",
    "œ":   "æ",
    "ʌ":   "a",
    "ɐ":   "a",
}

# Rengmitca inventory — imported lazily to avoid circular imports at module level
_INVENTORY: set[str] | None = None


def _get_inventory() -> set[str]:
    global _INVENTORY
    if _INVENTORY is None:
        from config import RENGMITCA_INVENTORY
        _INVENTORY = RENGMITCA_INVENTORY
    return _INVENTORY


@dataclass
class MappedPhone:
    source: str      # original IPA phone from G2P
    mapped: str      # phone in Rengmitca inventory (or '?')
    flagged: bool    # True if no good match was found


@dataclass
class MappingResult:
    phones: list[MappedPhone] = field(default_factory=list)
    tone: str = ""  # Tone annotation — empty by default, filled during review

    @property
    def ipa_string(self) -> str:
        """Space-separated IPA transcription using Rengmitca phones."""
        return " ".join(p.mapped for p in self.phones if p.mapped != "?")

    @property
    def ipa_string_with_flags(self) -> str:
        """IPA string with '?' markers for uncertain phones."""
        return " ".join(p.mapped for p in self.phones)

    @property
    def flagged_fraction(self) -> float:
        if not self.phones:
            return 0.0
        return sum(1 for p in self.phones if p.flagged) / len(self.phones)


def map_phones_to_rengmitca(ipa_phones: list[str]) -> MappingResult:
    """
    Map a list of IPA phones (from Mizo G2P) to the Rengmitca inventory.
    """
    inventory = _get_inventory()
    mapped_phones: list[MappedPhone] = []

    for phone in ipa_phones:
        if phone in inventory:
            mapped_phones.append(MappedPhone(source=phone, mapped=phone, flagged=False))
        elif phone in _NEAR_MATCH and _NEAR_MATCH[phone] in inventory:
            mapped_phones.append(MappedPhone(
                source=phone, mapped=_NEAR_MATCH[phone], flagged=False,
            ))
        else:
            mapped_phones.append(MappedPhone(source=phone, mapped="?", flagged=True))

    return MappingResult(phones=mapped_phones)


def text_to_rengmitca_ipa(mizo_text: str) -> MappingResult:
    """
    Full pipeline: Mizo Latin text → Mizo IPA → Rengmitca inventory.
    Words are separated by spaces; output phones span all words with a
    word-boundary marker omitted (phones only).
    """
    all_phones: list[str] = []
    for word in mizo_text.split():
        word_phones = mizo_to_ipa(word)
        all_phones.extend(word_phones)
    return map_phones_to_rengmitca(all_phones)


def map_aligned_phonemes_to_rengmitca(phoneme_tokens):
    """
    Map aligned phoneme tokens to Rengmitca inventory while preserving timing.
    
    Takes PhonemeToken objects (from align.py) with IPA phonemes and timing,
    maps them through the Rengmitca inventory substitution rules, and returns
    new PhonemeToken objects with Rengmitca phonemes.
    
    Parameters
    ----------
    phoneme_tokens : list of PhonemeToken (from pipeline.align)
                     Each has: phone (IPA), start, end, confidence
    
    Returns
    -------
    list of PhonemeToken with Rengmitca phonemes, preserving timing/confidence.
    Also returns flagged status for each phoneme (True if no good match).
    """
    from pipeline.align import PhonemeToken
    
    inventory = _get_inventory()
    mapped_tokens = []
    
    for pt in phoneme_tokens:
        ipa_phone = pt.phone
        
        # Check if phone is in Rengmitca inventory
        if ipa_phone in inventory:
            # Direct match - keep as-is
            mapped_tokens.append(PhonemeToken(
                phone=ipa_phone,
                start=pt.start,
                end=pt.end,
                confidence=pt.confidence,
            ))
        elif ipa_phone in _NEAR_MATCH and _NEAR_MATCH[ipa_phone] in inventory:
            # Near match - substitute
            rengmitca_phone = _NEAR_MATCH[ipa_phone]
            # Reduce confidence slightly for substitutions
            adjusted_conf = pt.confidence * 0.95
            mapped_tokens.append(PhonemeToken(
                phone=rengmitca_phone,
                start=pt.start,
                end=pt.end,
                confidence=round(adjusted_conf, 4),
            ))
        else:
            # No match - flag as uncertain with "?"
            mapped_tokens.append(PhonemeToken(
                phone="?",
                start=pt.start,
                end=pt.end,
                confidence=pt.confidence * 0.5,  # Heavily penalize confidence
            ))
    
    return mapped_tokens

