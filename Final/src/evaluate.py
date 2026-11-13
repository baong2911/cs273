"""Metrics, plots, error analysis, and coefficient extraction."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import ALL_LINGUISTIC_FEATURES


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute consistent binary metrics with positive class 1."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    path: Path | str,
    title: str,
) -> np.ndarray:
    """Save a labeled confusion-matrix image and return the matrix."""

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return matrix


def build_error_table(
    reviews: pd.DataFrame,
    y_true,
    y_pred,
    feature_set: str,
    classifier: str,
) -> pd.DataFrame:
    """Return incorrectly classified review rows for one final model."""

    required = {"review_id", "review", "rating", "label"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Review dataframe is missing columns: {sorted(missing)}")
    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)
    if len(reviews) != len(true_array) or len(true_array) != len(pred_array):
        raise ValueError("reviews, y_true, and y_pred must have equal lengths")
    incorrect = true_array != pred_array
    result = reviews.loc[incorrect, ["review_id", "review", "rating"]].copy()
    result["true_label"] = true_array[incorrect].astype(int)
    result["feature_set"] = feature_set
    result["classifier"] = classifier
    result["predicted_label"] = pred_array[incorrect].astype(int)
    return result[
        [
            "review_id",
            "review",
            "rating",
            "true_label",
            "feature_set",
            "classifier",
            "predicted_label",
        ]
    ]


def extract_linguistic_coefficients(
    model,
    feature_set: str,
    classifier: str,
) -> pd.DataFrame:
    """Extract only named linguistic coefficients from an interpretable model.

    For ``all15_doc2vec``, the first fifteen scaled coefficients correspond to
    the named linguistic dimensions; Doc2Vec dimensions are deliberately not
    assigned semantic labels.
    """

    empty_columns = [
        "feature_set", "classifier", "feature", "coefficient", "abs_coefficient"
    ]
    if feature_set == "doc2vec":
        return pd.DataFrame(columns=empty_columns)
    if feature_set == "basic10":
        feature_names = list(ALL_LINGUISTIC_FEATURES[:10])
    elif feature_set == "extra5":
        feature_names = list(ALL_LINGUISTIC_FEATURES[10:])
    elif feature_set in {"all15", "all15_doc2vec"}:
        feature_names = list(ALL_LINGUISTIC_FEATURES)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    try:
        coefficients = np.asarray(model.named_steps["classifier"].coef_)[0]
    except (AttributeError, KeyError, IndexError) as exc:
        raise ValueError("The supplied model does not expose linear coefficients") from exc
    coefficients = coefficients[: len(feature_names)]
    if len(coefficients) != len(feature_names):
        raise ValueError("Model coefficient count is incompatible with feature set")
    return pd.DataFrame(
        {
            "feature_set": feature_set,
            "classifier": classifier,
            "feature": feature_names,
            "coefficient": coefficients.astype(float),
            "abs_coefficient": np.abs(coefficients.astype(float)),
        }
    )
