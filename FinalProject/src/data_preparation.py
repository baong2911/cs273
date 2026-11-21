"""Load, clean, label, balance, tokenize, and split hotel reviews."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

try:  # Supports both ``python main.py`` and ``python src/data_preparation.py``.
    from .nlp_utils import sentence_tokenize, word_tokenize
except ImportError:  # pragma: no cover - only used for direct script execution
    from nlp_utils import sentence_tokenize, word_tokenize


DEFAULT_SAMPLE_PER_CLASS = 3_000
DEFAULT_RANDOM_STATE = 42

_REVIEW_CANDIDATES = (
    "review",
    "reviews",
    "review_text",
    "reviewtext",
    "text",
    "hotel_review",
)
_RATING_CANDIDATES = ("rating", "ratings", "star_rating", "stars", "score")


def _normalized_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).casefold()).strip("_")


def _find_column(columns: Iterable[object], candidates: Iterable[str], kind: str) -> object:
    """Find a likely CSV column and raise an actionable error if none exists."""

    column_list = list(columns)
    normalized = {_normalized_column_name(column): column for column in column_list}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    # TripAdvisor exports vary a little.  A conservative substring fallback
    # handles names such as ``Hotel_Review`` without silently choosing a
    # wholly unrelated column.
    for normalized_name, original_name in normalized.items():
        if kind == "review" and "review" in normalized_name:
            return original_name
        if kind == "rating" and ("rating" in normalized_name or "star" in normalized_name):
            return original_name

    available = ", ".join(map(str, column_list))
    expected = " or ".join(candidates)
    raise ValueError(
        f"Could not identify the {kind} column. Expected a column like {expected}; "
        f"found: {available}."
    )


def load_and_clean_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load a TripAdvisor CSV and remove invalid or duplicate review rows.

    The returned frame uses the stable canonical columns ``review`` and
    ``rating`` regardless of the source file's capitalization.  Duplicate
    detection is case-insensitive and treats repeated whitespace as identical.
    """

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Put tripadvisor_hotel_reviews.csv "
            "in the data directory or pass --data-path."
        )

    data = pd.read_csv(csv_path)
    review_column = _find_column(data.columns, _REVIEW_CANDIDATES, "review")
    rating_column = _find_column(data.columns, _RATING_CANDIDATES, "rating")

    cleaned = data[[review_column, rating_column]].rename(
        columns={review_column: "review", rating_column: "rating"}
    )
    cleaned = cleaned.dropna(subset=["review", "rating"]).copy()
    cleaned["review"] = cleaned["review"].astype(str).str.strip()
    cleaned = cleaned.loc[cleaned["review"].ne("")].copy()
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    cleaned = cleaned.dropna(subset=["rating"]).copy()

    normalized_reviews = (
        cleaned["review"].str.replace(r"\s+", " ", regex=True).str.casefold()
    )
    cleaned = cleaned.loc[~normalized_reviews.duplicated()].copy()
    return cleaned.reset_index(drop=True)


def create_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude neutral rows and label ratings 1--2/4--5 as 0/1 sentiment."""

    required = {"review", "rating"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Data frame is missing required column(s): {sorted(missing)}")

    labeled = df.copy()
    ratings = pd.to_numeric(labeled["rating"], errors="coerce")
    valid = ratings.isin([1, 2, 4, 5])
    labeled = labeled.loc[valid].copy()
    labeled["rating"] = ratings.loc[valid].astype(int)
    labeled["sentiment"] = (labeled["rating"] >= 4).astype(int)
    return labeled.reset_index(drop=True)


def create_balanced_sample(
    df: pd.DataFrame,
    sample_per_class: int = DEFAULT_SAMPLE_PER_CLASS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Return an exactly balanced random sample of negative and positive rows."""

    if "sentiment" not in df.columns:
        raise ValueError("Create sentiment labels before sampling the data.")
    if sample_per_class <= 0:
        raise ValueError("sample_per_class must be a positive integer.")

    counts = df["sentiment"].value_counts()
    missing_or_small = {
        label: int(counts.get(label, 0)) for label in (0, 1) if counts.get(label, 0) < sample_per_class
    }
    if missing_or_small:
        raise ValueError(
            "The cleaned dataset does not contain enough examples for the required "
            f"balanced sample of {sample_per_class} per class: {missing_or_small}."
        )

    negative = df.loc[df["sentiment"] == 0].sample(
        n=sample_per_class, random_state=random_state
    )
    positive = df.loc[df["sentiment"] == 1].sample(
        n=sample_per_class, random_state=random_state
    )
    sample = pd.concat([negative, positive], ignore_index=True)
    # Shuffle class blocks after independently sampling them, without changing
    # the reproducible seed used for selection.
    return sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def tokenize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Add NLTK word and sentence token lists needed by linguistic features."""

    if "review" not in df.columns:
        raise ValueError("Data frame is missing the required 'review' column.")
    tokenized = df.copy()
    reviews = tokenized["review"].fillna("").astype(str)
    tokenized["word_tokens"] = reviews.map(word_tokenize)
    tokenized["sentence_tokens"] = reviews.map(sentence_tokenize)
    return tokenized


def split_data(
    df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 60% train, 20% validation, and 20% test splits."""

    if "sentiment" not in df.columns:
        raise ValueError("Data frame is missing the required 'sentiment' column.")
    if df["sentiment"].nunique() != 2:
        raise ValueError("A stratified split requires both negative and positive reviews.")

    train, remaining = train_test_split(
        df,
        test_size=0.40,
        random_state=random_state,
        stratify=df["sentiment"],
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=random_state,
        stratify=remaining["sentiment"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def split_summary(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> Dict[str, Dict[str, int]]:
    """Return split counts for logging and reproducibility checks."""

    result: Dict[str, Dict[str, int]] = {}
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        counts = frame["sentiment"].value_counts()
        result[name] = {
            "total": int(len(frame)),
            "negative": int(counts.get(0, 0)),
            "positive": int(counts.get(1, 0)),
        }
    return result

