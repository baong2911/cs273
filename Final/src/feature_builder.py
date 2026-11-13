"""Build and persist the five canonical feature representations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    ALL_LINGUISTIC_FEATURES,
    BASIC_FEATURES,
    FEATURE_SET_NAMES,
    FEATURES_DIR,
)
from .linguistic_features import extract_linguistic_matrix


def _linguistic_array(linguistic_features) -> np.ndarray:
    if isinstance(linguistic_features, pd.DataFrame):
        missing = set(ALL_LINGUISTIC_FEATURES) - set(linguistic_features.columns)
        if missing:
            raise ValueError(f"Missing linguistic feature columns: {sorted(missing)}")
        array = linguistic_features.loc[:, list(ALL_LINGUISTIC_FEATURES)].to_numpy(dtype=float)
    else:
        array = np.asarray(linguistic_features, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(ALL_LINGUISTIC_FEATURES):
        raise ValueError(
            f"Linguistic features must have shape (n, {len(ALL_LINGUISTIC_FEATURES)}); "
            f"received {array.shape}"
        )
    return np.asarray(array, dtype=np.float64)


def _doc2vec_array(doc2vec_vectors, rows: int) -> np.ndarray:
    if doc2vec_vectors is None:
        raise ValueError("Doc2Vec vectors are required for doc2vec feature sets")
    array = np.asarray(doc2vec_vectors, dtype=float)
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(
            f"Doc2Vec vectors must have shape (n, d) with n={rows}; received {array.shape}"
        )
    return np.asarray(array, dtype=np.float64)


def build_feature_sets(
    texts: Sequence[str] | None,
    linguistic_features=None,
    doc2vec_vectors=None,
) -> dict[str, np.ndarray]:
    """Return all available canonical feature sets in stable key order.

    ``doc2vec_vectors`` is optional so linguistic-only prediction can reuse the
    builder without loading a Doc2Vec artifact. When it is omitted, the two
    Doc2Vec-containing keys are intentionally not returned.
    """

    if linguistic_features is None:
        if texts is None:
            raise ValueError("texts or linguistic_features must be supplied")
        linguistic_features = extract_linguistic_matrix(texts)
    linguistic = _linguistic_array(linguistic_features)
    if texts is not None and len(texts) != linguistic.shape[0]:
        raise ValueError("texts and linguistic_features must contain the same number of rows")

    basic_end = len(BASIC_FEATURES)
    sets: dict[str, np.ndarray] = {
        "basic10": linguistic[:, :basic_end],
        "extra5": linguistic[:, basic_end:],
        "all15": linguistic,
    }
    if doc2vec_vectors is not None:
        doc2vec = _doc2vec_array(doc2vec_vectors, linguistic.shape[0])
        sets["doc2vec"] = doc2vec
        sets["all15_doc2vec"] = np.hstack((linguistic, doc2vec))
    return sets


def save_feature_sets(
    feature_sets: Mapping[str, np.ndarray],
    split_name: str,
    output_dir: Path | str = FEATURES_DIR,
) -> None:
    """Save each supplied feature matrix under its canonical filename."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_set, matrix in feature_sets.items():
        if feature_set not in FEATURE_SET_NAMES:
            raise ValueError(f"Unknown feature set: {feature_set}")
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"Feature matrix must be two-dimensional: {feature_set}")
        np.save(output_dir / f"{split_name}_{feature_set}.npy", array)


def load_feature_set(
    split_name: str,
    feature_set: str,
    input_dir: Path | str = FEATURES_DIR,
) -> np.ndarray:
    """Load one generated feature matrix."""

    path = Path(input_dir) / f"{split_name}_{feature_set}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {path}")
    return np.asarray(np.load(path), dtype=np.float64)
