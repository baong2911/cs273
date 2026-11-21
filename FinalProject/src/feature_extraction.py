"""Linguistic feature extraction shared by training and the GUI.

Feature ordering is deliberately fixed by the public ``*_FEATURE_NAMES``
constants.  Models and the GUI therefore always use the same representation.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from spellchecker import SpellChecker
except ImportError:  # The dependency is listed in requirements.txt.
    SpellChecker = None  # type: ignore

try:
    from .nlp_utils import get_vader_analyzer, pos_tag, sentence_tokenize, word_tokenize
except ImportError:  # pragma: no cover - only used for direct script execution
    from nlp_utils import get_vader_analyzer, pos_tag, sentence_tokenize, word_tokenize


BASIC_FEATURE_NAMES: Tuple[str, ...] = (
    "average_word_length",
    "average_sentence_length",
    "word_count",
    "verb_count",
    "adjective_count",
    "passive_voice_count",
    "sentence_count",
    "content_word_diversity",
    "typo_count",
    "typo_ratio",
)

ADDITIONAL_FEATURE_NAMES: Tuple[str, ...] = (
    "negation_count",
    "first_person_pronoun_ratio",
    "adverb_count",
    "exclamation_mark_count",
    "sentiment_lexicon_score",
)

ALL_FEATURE_NAMES: Tuple[str, ...] = BASIC_FEATURE_NAMES + ADDITIONAL_FEATURE_NAMES

_WORD_RE = re.compile(r"^[A-Za-z]+(?:['’][A-Za-z]+)?$")
_PASSIVE_AUXILIARIES = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "is",
    "was",
    "were",
}
_NEGATIONS = {
    "ain't",
    "barely",
    "cannot",
    "can't",
    "didn't",
    "doesn't",
    "don't",
    "hardly",
    "isn't",
    "neither",
    "never",
    "no",
    "nobody",
    "none",
    "nor",
    "not",
    "nothing",
    "nowhere",
    "n't",
    "shouldn't",
    "wasn't",
    "weren't",
    "without",
    "wouldn't",
}
_FIRST_PERSON_PRONOUNS = {
    "i",
    "i'm",
    "i've",
    "i'll",
    "i'd",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "we're",
    "we've",
    "we'll",
    "we'd",
    "us",
    "our",
    "ours",
    "ourselves",
}
# This is used only if NLTK's VADER data has not been installed.  VADER is
# always preferred and is the score used in a normal project run.
_FALLBACK_POSITIVE = {
    "amazing",
    "beautiful",
    "clean",
    "comfortable",
    "excellent",
    "fantastic",
    "friendly",
    "good",
    "great",
    "love",
    "nice",
    "perfect",
    "recommend",
    "wonderful",
}
_FALLBACK_NEGATIVE = {
    "awful",
    "bad",
    "dirty",
    "disappointing",
    "hate",
    "horrible",
    "poor",
    "rude",
    "terrible",
    "uncomfortable",
    "worst",
}


def _as_text(review: object) -> str:
    return "" if review is None else str(review)


def _is_word(token: str) -> bool:
    return bool(_WORD_RE.match(token))


def _review_words(text: str) -> List[str]:
    return [token for token in word_tokenize(text) if _is_word(token)]


@lru_cache(maxsize=1)
def _get_spellchecker():
    if SpellChecker is None:
        return None
    return SpellChecker()


@lru_cache(maxsize=100_000)
def _is_typo(word: str) -> bool:
    """Return whether one alphabetic token is unknown to pyspellchecker."""

    checker = _get_spellchecker()
    if checker is None or len(word) <= 1:
        return False
    try:
        return word.casefold() in checker.unknown([word.casefold()])
    except (AttributeError, TypeError):
        # A feature extractor should remain usable if an alternate spellchecker
        # implementation is installed, rather than fail an entire experiment.
        return False


def _count_passive_voice(tagged_words: Sequence[Tuple[str, str]]) -> int:
    """Count common ``be + past participle`` passive constructions.

    This intentionally uses a transparent POS-pattern heuristic rather than
    claiming full syntactic parsing.  It catches forms such as "was cleaned"
    and "were very poorly maintained".
    """

    count = 0
    for index, (word, _tag) in enumerate(tagged_words):
        if word.casefold() not in _PASSIVE_AUXILIARIES:
            continue
        look_ahead = index + 1
        # Permit intervening adverbs and "not" but do not join across a long
        # phrase or sentence.
        while look_ahead < len(tagged_words) and look_ahead <= index + 3:
            next_word, next_tag = tagged_words[look_ahead]
            if next_tag.startswith("RB") or next_word.casefold() in _NEGATIONS:
                look_ahead += 1
                continue
            if next_tag == "VBN":
                count += 1
            break
    return count


def _sentiment_lexicon_score(text: str, words: Sequence[str]) -> float:
    analyzer = get_vader_analyzer()
    if analyzer is not None:
        return float(analyzer.polarity_scores(text)["compound"])

    lower_words = [word.casefold() for word in words]
    positive = sum(word in _FALLBACK_POSITIVE for word in lower_words)
    negative = sum(word in _FALLBACK_NEGATIVE for word in lower_words)
    return float((positive - negative) / max(len(lower_words), 1))


def _analyze_review(review: object) -> Dict[str, object]:
    text = _as_text(review)
    words = _review_words(text)
    sentences = sentence_tokenize(text)
    tagged_words = pos_tag(words)
    return {
        "text": text,
        "words": words,
        "sentences": sentences,
        "tagged_words": tagged_words,
    }


def _basic_from_analysis(analysis: Mapping[str, object]) -> np.ndarray:
    text = str(analysis["text"])
    words = list(analysis["words"])  # type: ignore[arg-type]
    sentences = list(analysis["sentences"])  # type: ignore[arg-type]
    tagged_words = list(analysis["tagged_words"])  # type: ignore[arg-type]
    word_count = len(words)

    average_word_length = (
        float(sum(len(word) for word in words) / word_count) if word_count else 0.0
    )
    sentence_count = len(sentences)
    if sentence_count:
        words_per_sentence = [len(_review_words(sentence)) for sentence in sentences]
        average_sentence_length = float(sum(words_per_sentence) / sentence_count)
    else:
        average_sentence_length = 0.0

    verb_count = sum(tag.startswith("VB") for _word, tag in tagged_words)
    adjective_count = sum(tag.startswith("JJ") for _word, tag in tagged_words)
    passive_voice_count = _count_passive_voice(tagged_words)
    content_words = [
        word.casefold()
        for word, tag in tagged_words
        if tag.startswith(("NN", "VB", "JJ", "RB"))
    ]
    content_word_diversity = (
        float(len(set(content_words)) / len(content_words)) if content_words else 0.0
    )
    typo_count = sum(_is_typo(word) for word in words)
    typo_ratio = float(typo_count / word_count) if word_count else 0.0

    return np.asarray(
        [
            average_word_length,
            average_sentence_length,
            float(word_count),
            float(verb_count),
            float(adjective_count),
            float(passive_voice_count),
            float(sentence_count),
            content_word_diversity,
            float(typo_count),
            typo_ratio,
        ],
        dtype=float,
    )


def _additional_from_analysis(analysis: Mapping[str, object]) -> np.ndarray:
    text = str(analysis["text"])
    words = list(analysis["words"])  # type: ignore[arg-type]
    tagged_words = list(analysis["tagged_words"])  # type: ignore[arg-type]
    lower_words = [word.casefold() for word in words]
    word_count = len(words)

    negation_count = sum(word in _NEGATIONS for word in lower_words)
    first_person_count = sum(word in _FIRST_PERSON_PRONOUNS for word in lower_words)
    first_person_ratio = float(first_person_count / word_count) if word_count else 0.0
    adverb_count = sum(tag.startswith("RB") for _word, tag in tagged_words)
    sentiment_score = _sentiment_lexicon_score(text, words)

    return np.asarray(
        [
            float(negation_count),
            first_person_ratio,
            float(adverb_count),
            float(text.count("!")),
            sentiment_score,
        ],
        dtype=float,
    )


def extract_basic_features(review: object) -> np.ndarray:
    """Return the ten basic linguistic features in ``BASIC_FEATURE_NAMES`` order."""

    return _basic_from_analysis(_analyze_review(review))


def extract_additional_features(review: object) -> np.ndarray:
    """Return the five additional linguistic features in fixed documented order."""

    return _additional_from_analysis(_analyze_review(review))


def extract_all_features(review: object) -> np.ndarray:
    """Return all fifteen linguistic features without duplicating NLP work."""

    analysis = _analyze_review(review)
    return np.concatenate((_basic_from_analysis(analysis), _additional_from_analysis(analysis)))


def _review_list(reviews: Iterable[object]) -> List[object]:
    if isinstance(reviews, str):
        return [reviews]
    return list(reviews)


def build_feature_matrices(reviews: Iterable[object]) -> Dict[str, np.ndarray]:
    """Build basic, additional, and all-linguistic matrices for *reviews*.

    The dictionary keys intentionally mirror the feature groups used later by
    ``train_models.build_feature_sets``:
    ``basic``, ``additional``, and ``all_linguistic``.
    """

    review_list = _review_list(reviews)
    if not review_list:
        all_features = np.empty((0, len(ALL_FEATURE_NAMES)), dtype=float)
    else:
        all_features = np.vstack([extract_all_features(review) for review in review_list])
    return {
        "basic": all_features[:, : len(BASIC_FEATURE_NAMES)],
        "additional": all_features[:, len(BASIC_FEATURE_NAMES) :],
        "all_linguistic": all_features,
    }

