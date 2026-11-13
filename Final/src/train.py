"""Scikit-learn model construction and persistence."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import RANDOM_SEED


def normalize_classifier_name(classifier_name: str) -> str:
    """Normalize a human-readable classifier name to ``logistic`` or ``svm``."""

    normalized = classifier_name.strip().casefold().replace(" ", "_")
    aliases = {
        "logistic": "logistic",
        "logistic_regression": "logistic",
        "logisticregression": "logistic",
        "svm": "svm",
        "linear_svm": "svm",
        "linearsvm": "svm",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return aliases[normalized]


def make_pipeline(classifier_name: str, C: float = 1.0) -> Pipeline:
    """Create the required StandardScaler plus linear classifier pipeline."""

    classifier_name = normalize_classifier_name(classifier_name)
    if C <= 0:
        raise ValueError("C must be positive")
    if classifier_name == "logistic":
        classifier = LogisticRegression(
            C=float(C),
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_SEED,
        )
    else:
        classifier = LinearSVC(
            C=float(C),
            max_iter=5000,
            random_state=RANDOM_SEED,
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def fit_model(X, y, classifier_name: str, C: float = 1.0) -> Pipeline:
    """Fit and return a classifier pipeline."""

    normalized_name = normalize_classifier_name(classifier_name)
    model = make_pipeline(normalized_name, C=C)
    model.fit(X, y)
    # Metadata is harmless for joblib and makes manually loaded artifacts more
    # self-describing while the filename remains the canonical routing source.
    model.classifier_name_ = normalized_name
    model.C_value_ = float(C)
    return model


def save_model(model: Pipeline, path: Path | str) -> None:
    """Persist a fitted pipeline."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str) -> Pipeline:
    """Load a persisted fitted pipeline."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)
