"""Loading, cleaning, sampling, and splitting the TripAdvisor data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    BALANCED_DATA_PATH,
    LABEL_COLUMN,
    LABEL_MAPPING,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RATING_COLUMN,
    RAW_DATA_PATH,
    RAW_RATING_COLUMN,
    RAW_REVIEW_COLUMN,
    REVIEW_COLUMN,
    REVIEW_ID_COLUMN,
    SAMPLE_PER_CLASS,
    TEST_DATA_PATH,
    TEST_FRACTION,
    TRAIN_DATA_PATH,
    TRAIN_FRACTION,
    VALIDATION_DATA_PATH,
    VALIDATION_FRACTION,
)


SAVED_COLUMNS = [REVIEW_ID_COLUMN, REVIEW_COLUMN, RATING_COLUMN, LABEL_COLUMN]


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle CSV and validate its required columns."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download the Kaggle CSV and place it there."
        )
    frame = pd.read_csv(path)
    missing = {RAW_REVIEW_COLUMN, RAW_RATING_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Raw dataset is missing required column(s): {', '.join(sorted(missing))}"
        )
    return frame


def _normalize_review_for_dedup(review: str) -> str:
    """Normalize only enough to identify duplicate review text."""

    # Keep punctuation and spelling. Case-folding and whitespace normalization
    # make duplicate detection stable without changing the text used as input.
    return re.sub(r"\s+", " ", str(review).strip()).casefold()


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw rows and assign the binary sentiment label.

    The returned frame uses the project's canonical lower-case column names and
    contains no 3-star, missing, blank, invalid-rating, or duplicate review.
    """

    missing = {RAW_REVIEW_COLUMN, RAW_RATING_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"Input dataframe is missing required column(s): {', '.join(sorted(missing))}"
        )

    work = df[[RAW_REVIEW_COLUMN, RAW_RATING_COLUMN]].copy()
    work = work.rename(
        columns={RAW_REVIEW_COLUMN: REVIEW_COLUMN, RAW_RATING_COLUMN: RATING_COLUMN}
    )
    work = work.dropna(subset=[REVIEW_COLUMN, RATING_COLUMN])
    work[REVIEW_COLUMN] = work[REVIEW_COLUMN].astype(str).str.strip()
    work = work[work[REVIEW_COLUMN].ne("")]
    work[RATING_COLUMN] = pd.to_numeric(work[RATING_COLUMN], errors="coerce")
    work = work.dropna(subset=[RATING_COLUMN])
    non_integer_ratings = work[RATING_COLUMN].mod(1).ne(0)
    if non_integer_ratings.any():
        invalid = sorted(work.loc[non_integer_ratings, RATING_COLUMN].unique().tolist())
        raise ValueError(f"Ratings must be integers; found {invalid}")
    work[RATING_COLUMN] = work[RATING_COLUMN].astype(int)

    work["_normalized_review"] = work[REVIEW_COLUMN].map(_normalize_review_for_dedup)
    work = work.drop_duplicates(subset=["_normalized_review"], keep="first")
    work = work[work[RATING_COLUMN].ne(3)]

    invalid_ratings = set(work[RATING_COLUMN].unique()) - set(LABEL_MAPPING)
    if invalid_ratings:
        raise ValueError(
            "Ratings must be in {1, 2, 4, 5} after removing 3-star reviews; "
            f"found {sorted(invalid_ratings)}"
        )

    work[LABEL_COLUMN] = work[RATING_COLUMN].map(LABEL_MAPPING).astype(int)
    work = work.drop(columns=["_normalized_review"]).reset_index(drop=True)
    work.insert(0, REVIEW_ID_COLUMN, np.arange(len(work), dtype=int))
    return work[SAVED_COLUMNS]


def create_balanced_sample(
    df: pd.DataFrame, per_class: int = SAMPLE_PER_CLASS, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Sample exactly ``per_class`` examples for each binary label."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    required = {REVIEW_COLUMN, RATING_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing columns: {sorted(missing)}")
    work = df.copy()
    if REVIEW_ID_COLUMN not in work.columns:
        work.insert(0, REVIEW_ID_COLUMN, np.arange(len(work), dtype=int))

    counts = work[LABEL_COLUMN].value_counts().to_dict()
    insufficient = {
        label: int(counts.get(label, 0))
        for label in (NEGATIVE_LABEL, POSITIVE_LABEL)
        if counts.get(label, 0) < per_class
    }
    if insufficient:
        raise ValueError(
            f"Not enough examples for balanced sampling: need {per_class} per class, "
            f"available {insufficient}"
        )

    pieces = [
        work[work[LABEL_COLUMN].eq(label)].sample(
            n=per_class, random_state=seed, replace=False
        )
        for label in (NEGATIVE_LABEL, POSITIVE_LABEL)
    ]
    sampled = pd.concat(pieces, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled[SAVED_COLUMNS]


def split_dataset(
    df: pd.DataFrame, seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a reproducible stratified 60/20/20 train/validation/test split."""

    required = {REVIEW_COLUMN, RATING_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing columns: {sorted(missing)}")
    work = df.copy()
    if REVIEW_ID_COLUMN not in work.columns:
        work.insert(0, REVIEW_ID_COLUMN, np.arange(len(work), dtype=int))
    if not np.isclose(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION, 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1")

    train_validation, test = train_test_split(
        work,
        test_size=TEST_FRACTION,
        random_state=seed,
        stratify=work[LABEL_COLUMN],
    )
    validation_relative = VALIDATION_FRACTION / (TRAIN_FRACTION + VALIDATION_FRACTION)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_relative,
        random_state=seed,
        stratify=train_validation[LABEL_COLUMN],
    )
    return (
        train.reset_index(drop=True)[SAVED_COLUMNS],
        validation.reset_index(drop=True)[SAVED_COLUMNS],
        test.reset_index(drop=True)[SAVED_COLUMNS],
    )


def save_splits(
    balanced: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path | str = PROCESSED_DATA_DIR,
) -> None:
    """Write the balanced sample and split CSVs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    balanced[SAVED_COLUMNS].to_csv(output_dir / BALANCED_DATA_PATH.name, index=False)
    train[SAVED_COLUMNS].to_csv(output_dir / TRAIN_DATA_PATH.name, index=False)
    validation[SAVED_COLUMNS].to_csv(
        output_dir / VALIDATION_DATA_PATH.name, index=False
    )
    test[SAVED_COLUMNS].to_csv(output_dir / TEST_DATA_PATH.name, index=False)


def prepare_dataset(
    raw_path: Path | str = RAW_DATA_PATH,
    per_class: int = SAMPLE_PER_CLASS,
    seed: int = RANDOM_SEED,
    output_dir: Path | str = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete preparation workflow and save all outputs."""

    cleaned = clean_and_label(load_raw_data(raw_path))
    balanced = create_balanced_sample(cleaned, per_class=per_class, seed=seed)
    train, validation, test = split_dataset(balanced, seed=seed)
    save_splits(balanced, train, validation, test, output_dir=output_dir)
    return balanced, train, validation, test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--per-class", type=int, default=SAMPLE_PER_CLASS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    balanced, train, validation, test = prepare_dataset(
        raw_path=args.input,
        per_class=args.per_class,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(
        f"Prepared {len(balanced):,} reviews: "
        f"train={len(train):,}, validation={len(validation):,}, test={len(test):,}"
    )
    print("Label counts by split:")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        print(f"  {name}: {frame[LABEL_COLUMN].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
