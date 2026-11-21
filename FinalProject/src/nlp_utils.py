"""Small, shared NLTK helpers used by data preparation and features.

The project is intended to use NLTK's sentence tokenizer, word tokenizer,
part-of-speech tagger, and VADER sentiment lexicon.  The helpers below use
those resources whenever they are installed.  A lightweight NLTK-based
fallback keeps the program usable in restricted environments, while the
README explains how to install the full resources for the intended results.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Dict, List, Sequence, Tuple

import nltk
from nltk.tokenize import wordpunct_tokenize


# The alternate punkt/tagger resource names are needed by different supported
# NLTK releases.  ``ensure_nltk_resources`` checks and optionally downloads
# both so a fresh installation works without version-specific instructions.
NLTK_RESOURCES: Tuple[Tuple[str, str], ...] = (
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    (
        "averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger_eng",
    ),
    ("vader_lexicon", "sentiment/vader_lexicon.zip"),
)

_FALLBACK_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def ensure_nltk_resources(download: bool = False) -> Dict[str, bool]:
    """Return installed-resource status, optionally downloading missing data.

    Downloading is explicit so importing the project never unexpectedly makes
    a network request.  ``main.py --download-nltk`` is the convenient way to
    enable it on a new machine.
    """

    status: Dict[str, bool] = {}
    for package_name, resource_path in NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
            status[package_name] = True
            continue
        except LookupError:
            pass

        if download:
            nltk.download(package_name, quiet=True)

        try:
            nltk.data.find(resource_path)
            status[package_name] = True
        except LookupError:
            status[package_name] = False
    return status


def sentence_tokenize(text: object) -> List[str]:
    """Tokenize *text* into sentences using NLTK, with a safe fallback."""

    normalized = "" if text is None else str(text).strip()
    if not normalized:
        return []
    try:
        return [sentence for sentence in nltk.sent_tokenize(normalized) if sentence.strip()]
    except LookupError:
        # ``wordpunct_tokenize`` below is still NLTK.  This only avoids an
        # opaque crash when a student has not yet downloaded punkt data.
        return [part for part in _FALLBACK_SENTENCE_RE.split(normalized) if part.strip()]


def word_tokenize(text: object) -> List[str]:
    """Tokenize *text* into word/punctuation tokens using NLTK."""

    normalized = "" if text is None else str(text).strip()
    if not normalized:
        return []
    try:
        return nltk.word_tokenize(normalized)
    except LookupError:
        return wordpunct_tokenize(normalized)


def pos_tag(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    """POS-tag tokens with NLTK, falling back to conservative heuristics."""

    token_list = list(tokens)
    if not token_list:
        return []
    try:
        return nltk.pos_tag(token_list)
    except LookupError:
        return [(token, _heuristic_pos_tag(token)) for token in token_list]


def _heuristic_pos_tag(token: str) -> str:
    """Return a minimal fallback POS tag when NLTK tagger data is absent."""

    lower = token.casefold()
    if lower in {
        "am",
        "are",
        "be",
        "been",
        "being",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "was",
        "were",
    } or lower.endswith(("ed", "ing", "ize", "ise")):
        return "VB"
    if lower.endswith("ly") or lower in {"not", "never", "very", "too"}:
        return "RB"
    if lower.endswith(("able", "al", "ful", "ic", "ive", "less", "ous")):
        return "JJ"
    return "NN"


@lru_cache(maxsize=1)
def get_vader_analyzer():
    """Return NLTK's VADER analyzer, or ``None`` if its data is unavailable."""

    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except LookupError:
        return None

