"""Extraction of the project's fifteen hand-engineered linguistic features."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

from .config import ALL_LINGUISTIC_FEATURES
from .text_processing import (
    content_word_tag,
    is_alphabetic_token,
    lemmatize_token,
    pos_tag_tokens,
    sentence_tokenize,
    word_tokenize,
    word_tokens,
)


NEGATION_WORDS = {"no", "not", "never", "none", "neither", "nor"}
FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
}
BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being", "'m", "'s"}


@lru_cache(maxsize=1)
def _spell_checker():
    try:
        from spellchecker import SpellChecker
    except ImportError:  # pragma: no cover - only used in minimal environments
        return None
    return SpellChecker(language="en")


FALLBACK_DICTIONARY = {
    "a", "am", "an", "and", "are", "awful", "bad", "be", "beautiful", "by",
    "clean", "cleaned", "comfortable", "do", "excellent", "feel", "for", "great",
    "happy", "hate", "hotel", "i", "in", "is", "it", "location", "me", "my",
    "not", "of", "on", "poor", "room", "service", "staff", "stay", "terrible",
    "the", "this", "to", "very", "was", "we", "were", "wonderful",
}


@lru_cache(maxsize=1)
def _vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except (ImportError, LookupError):  # pragma: no cover - resource-dependent
        return None


def _fallback_sentiment_score(text: str) -> float:
    """Small deterministic fallback used only if the VADER corpus is absent."""

    positive = {
        "excellent", "great", "good", "love", "loved", "amazing", "clean", "friendly", "happy"
    }
    negative = {
        "terrible", "bad", "hate", "hated", "awful", "dirty", "rude", "poor", "disappointed"
    }
    tokens = {token.casefold() for token in word_tokens(text)}
    raw = len(tokens & positive) - len(tokens & negative)
    return float(np.clip(raw / 4.0, -1.0, 1.0))


def _sentiment_lexicon_score(text: str) -> float:
    vader = _vader()
    if vader is None:
        return _fallback_sentiment_score(text)
    return float(vader.polarity_scores(text)["compound"])


def _passive_voice_count(text: str) -> int:
    """Count sentence-level ``be + VBN`` patterns.

    This is intentionally a documented heuristic rather than a full syntactic
    parser: a form of *be* followed immediately by a past participle (or by up
    to two adverbs and then a ``VBN`` tag) counts as one likely passive pattern.
    """

    count = 0
    for sentence in sentence_tokenize(text):
        tagged = pos_tag_tokens(sentence)
        for index, (token, _tag) in enumerate(tagged):
            if token.casefold() not in BE_FORMS:
                continue
            look_ahead = index + 1
            skipped_adverbs = 0
            while (
                look_ahead < len(tagged)
                and tagged[look_ahead][1].startswith("RB")
                and skipped_adverbs < 2
            ):
                skipped_adverbs += 1
                look_ahead += 1
            if look_ahead < len(tagged) and tagged[look_ahead][1] == "VBN":
                count += 1
    return count


def extract_linguistic_features(text: str) -> dict[str, float]:
    """Extract all fifteen features in the canonical project order."""

    if text is None:
        text = ""
    text = str(text)
    tokens = word_tokenize(text)
    words = word_tokens(tokens)
    alphabetic_words = [token for token in words if is_alphabetic_token(token)]
    sentences = sentence_tokenize(text)

    word_count = len(words)
    sentence_count = len(sentences)
    average_word_length = (
        sum(len(re.sub(r"[^A-Za-z]", "", token)) for token in words) / word_count
        if word_count
        else 0.0
    )

    sentence_lengths = [len(word_tokens(sentence)) for sentence in sentences]
    average_sentence_length = (
        sum(sentence_lengths) / sentence_count if sentence_count else 0.0
    )

    tagged_tokens = pos_tag_tokens(tokens)
    tagged_words = [(token, tag) for token, tag in tagged_tokens if token in words]
    verb_count = sum(tag.startswith("VB") for _token, tag in tagged_words)
    adjective_count = sum(tag.startswith("JJ") for _token, tag in tagged_words)
    adverb_count = sum(tag.startswith("RB") for _token, tag in tagged_words)

    content_lemmas = [
        lemmatize_token(token, tag)
        for token, tag in tagged_words
        if content_word_tag(tag)
    ]
    content_word_diversity = (
        len(set(content_lemmas)) / len(content_lemmas) if content_lemmas else 0.0
    )

    # SpellChecker operates on alphabetic tokens. The dictionary is loaded once
    # and the original spelling is otherwise left untouched.
    checker = _spell_checker()
    lower_alphabetic_words = [token.casefold() for token in alphabetic_words]
    if checker is None:
        typo_count = sum(
            1 for token in lower_alphabetic_words if token not in FALLBACK_DICTIONARY
        )
    else:
        typo_count = sum(1 for token in lower_alphabetic_words if token not in checker)

    normalized_tokens = [token.casefold() for token in words]
    negation_count = sum(
        token in NEGATION_WORDS or token == "n't" or token.endswith("n't")
        for token in normalized_tokens
    )
    first_person_count = sum(token in FIRST_PERSON_PRONOUNS for token in normalized_tokens)

    features = {
        "avg_word_length": float(average_word_length),
        "avg_sentence_length": float(average_sentence_length),
        "word_count": float(word_count),
        "verb_count": float(verb_count),
        "adjective_count": float(adjective_count),
        "passive_voice_count": float(_passive_voice_count(text)),
        "sentence_count": float(sentence_count),
        "content_word_diversity": float(content_word_diversity),
        "typo_count": float(typo_count),
        "typo_ratio": float(typo_count / word_count if word_count else 0.0),
        "negation_count": float(negation_count),
        "first_person_pronoun_ratio": float(first_person_count / word_count if word_count else 0.0),
        "adverb_count": float(adverb_count),
        "exclamation_count": float(text.count("!")),
        "sentiment_lexicon_score": float(_sentiment_lexicon_score(text)),
    }
    # Protect downstream scalers from accidental infinities if a future feature
    # implementation changes.
    return {
        name: float(value) if np.isfinite(value) else 0.0
        for name, value in ((name, features.get(name, 0.0)) for name in ALL_LINGUISTIC_FEATURES)
    }


def extract_linguistic_matrix(texts: Sequence[str]) -> pd.DataFrame:
    """Extract a row-preserving DataFrame with the canonical 15 columns."""

    rows = [extract_linguistic_features(text) for text in texts]
    matrix = pd.DataFrame(rows, columns=list(ALL_LINGUISTIC_FEATURES), dtype=float)
    matrix = matrix.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return matrix
