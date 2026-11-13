"""Shared tokenization, POS-tagging, and lemmatization helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence

try:
    import nltk
except ImportError:  # pragma: no cover - only used in minimal environments
    nltk = None


_WORD_WITH_APOSTROPHE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'”’\)\]]*)\s+")


def _fallback_word_tokenize(text: str) -> list[str]:
    """A small tokenizer used only when Punkt resources are unavailable."""

    # Preserve n't as a separate token so the negation definition handles
    # contractions such as "didn't" consistently with NLTK's Treebank tokenizer.
    tokens: list[str] = []
    for token in _WORD_WITH_APOSTROPHE.findall(text):
        lower = token.casefold()
        if lower.endswith("n't") and len(token) > 3:
            tokens.append(token[:-3])
            tokens.append("n't")
        else:
            tokens.append(token)
    return tokens


def sentence_tokenize(text: str) -> list[str]:
    """Tokenize sentences, with a deterministic regex fallback."""

    if not text or not text.strip():
        return []
    try:
        if nltk is None:
            raise LookupError("NLTK is not installed")
        return [sentence for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    except LookupError:
        parts = _SENTENCE_BOUNDARY.split(text.strip())
        return [part.strip() for part in parts if part.strip()]


def word_tokenize(text: str) -> list[str]:
    """Tokenize words and punctuation using NLTK when available."""

    if not text:
        return []
    try:
        if nltk is None:
            raise LookupError("NLTK is not installed")
        return nltk.word_tokenize(text)
    except LookupError:
        return _fallback_word_tokenize(text)


def is_word_token(token: str) -> bool:
    """Return whether a token contains a word rather than only punctuation."""

    return bool(re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)?", token))


def is_alphabetic_token(token: str) -> bool:
    """Return whether a token consists only of alphabetic characters."""

    return bool(token) and token.isalpha()


def word_tokens(text_or_tokens: str | Sequence[str]) -> list[str]:
    """Return word tokens while preserving contractions when tokenized as one."""

    tokens = (
        word_tokenize(text_or_tokens)
        if isinstance(text_or_tokens, str)
        else list(text_or_tokens)
    )
    return [token for token in tokens if is_word_token(token)]


def _heuristic_pos_tag(tokens: Sequence[str]) -> list[tuple[str, str]]:
    """Conservative POS fallback for environments without a tagger corpus."""

    common_verbs = {
        "be", "am", "is", "are", "was", "were", "been", "being", "have",
        "has", "had", "do", "does", "did", "go", "went", "get", "got",
        "love", "loved", "like", "liked", "hate", "hated", "enjoy", "enjoyed",
        "stay", "stayed", "sleep", "slept", "visit", "visited", "feel", "felt",
        "look", "looked", "make", "made", "recommend", "recommended", "clean",
        "cleaned", "provide", "provided", "include", "included",
    }
    common_adjectives = {
        "good", "great", "bad", "terrible", "beautiful", "clean", "dirty", "friendly",
        "helpful", "comfortable", "quiet", "loud", "excellent", "poor", "amazing",
        "small", "large", "nice", "old", "new", "happy", "unhappy", "disappointing",
    }
    common_adverbs = {"very", "really", "quite", "too", "not", "never", "always", "quickly"}
    result: list[tuple[str, str]] = []
    for token in tokens:
        lower = token.casefold()
        if not is_word_token(token):
            result.append((token, "."))
        elif lower in common_adverbs or lower.endswith("ly"):
            result.append((token, "RB"))
        elif lower in common_adjectives or lower.endswith(("ful", "ous", "ive", "able", "less")):
            result.append((token, "JJ"))
        elif lower in common_verbs or lower.endswith(("ed", "ing")):
            result.append((token, "VBN" if lower.endswith("ed") else "VBG"))
        elif lower.endswith("s") and len(lower) > 3:
            result.append((token, "VBZ"))
        else:
            result.append((token, "NN"))
    return result


def pos_tag_tokens(text_or_tokens: str | Sequence[str]) -> list[tuple[str, str]]:
    """POS-tag tokens, falling back to simple suffix/lexicon rules."""

    tokens = (
        word_tokenize(text_or_tokens)
        if isinstance(text_or_tokens, str)
        else list(text_or_tokens)
    )
    if not tokens:
        return []
    try:
        if nltk is None:
            raise LookupError("NLTK is not installed")
        return nltk.pos_tag(tokens)
    except LookupError:
        return _heuristic_pos_tag(tokens)


def penn_to_wordnet_pos(tag: str) -> str | None:
    """Map a Penn Treebank tag to a WordNet lemmatizer category."""

    if tag.startswith("N"):
        return "n"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return None


@lru_cache(maxsize=1)
def _get_lemmatizer():
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


def lemmatize_token(token: str, tag: str | None = None) -> str:
    """Lemmatize a token when WordNet is available, otherwise lowercase it."""

    lower = token.casefold()
    try:
        lemmatizer = _get_lemmatizer()
        wordnet_pos = penn_to_wordnet_pos(tag or "")
        return (
            lemmatizer.lemmatize(lower, pos=wordnet_pos)
            if wordnet_pos
            else lemmatizer.lemmatize(lower)
        )
    except (ImportError, LookupError):
        return lower


def content_word_tag(tag: str) -> bool:
    """Return whether a POS tag is a noun, verb, adjective, or adverb."""

    return tag.startswith(("NN", "VB", "JJ", "RB"))


def model_tokens(text: str) -> list[str]:
    """Return lowercased word tokens suitable for Doc2Vec."""

    tokens = [token.casefold() for token in word_tokens(text)]
    return tokens or ["<empty>"]
