"""Final held-out-test evaluation, confusion matrices, and error analysis."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .feature_extraction import ALL_FEATURE_NAMES
    from .train_models import (
        CLASSIFIER_DISPLAY_NAMES,
        FEATURE_SET_DISPLAY_NAMES,
        iter_model_artifacts,
    )
except ImportError:  # pragma: no cover - only used for direct script execution
    from feature_extraction import ALL_FEATURE_NAMES
    from train_models import (
        CLASSIFIER_DISPLAY_NAMES,
        FEATURE_SET_DISPLAY_NAMES,
        iter_model_artifacts,
    )


def _label_name(label: int) -> str:
    return "Positive" if int(label) == 1 else "Negative"


def evaluate_model(model: object, X_test: object, y_test: Iterable[int]) -> Dict[str, object]:
    """Evaluate one fitted classifier on untouched test data.

    Returns scalar metrics, predictions, and a 2x2 confusion matrix ordered as
    ``[[true negative, false positive], [false negative, true positive]]``.
    """

    matrix = np.asarray(X_test, dtype=float)
    labels = np.asarray(list(y_test), dtype=int)
    if matrix.ndim != 2:
        raise ValueError(f"X_test must be two-dimensional, got shape {matrix.shape}.")
    if matrix.shape[0] != len(labels):
        raise ValueError("X_test and y_test have different numbers of rows.")
    if set(np.unique(labels)).difference({0, 1}):
        raise ValueError("Test labels must be 0 (negative) or 1 (positive).")
    if not hasattr(model, "predict"):
        raise TypeError("Model must provide a fitted predict() method.")

    predictions = np.asarray(model.predict(matrix), dtype=int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]),
    }


def evaluate_all_models(
    models: Mapping[str, object],
    feature_sets: Mapping[str, Mapping[str, object]],
    y_test: Iterable[int],
) -> Dict[str, object]:
    """Evaluate all ten saved classifier/feature-set combinations exactly once."""

    test_labels = np.asarray(list(y_test), dtype=int)
    rows: List[Dict[str, object]] = []
    predictions: Dict[str, np.ndarray] = {}
    matrices: Dict[str, np.ndarray] = {}

    for artifact in iter_model_artifacts(models):
        model_id = str(artifact["model_id"])
        classifier_id = str(artifact["classifier"])
        feature_set_id = str(artifact["feature_set"])
        if feature_set_id not in feature_sets:
            raise ValueError(f"No test feature matrix found for '{feature_set_id}'.")
        split_matrices = feature_sets[feature_set_id]
        if "test" not in split_matrices:
            raise ValueError(f"Feature set '{feature_set_id}' is missing a test matrix.")

        result = evaluate_model(artifact["estimator"], split_matrices["test"], test_labels)
        predictions[model_id] = np.asarray(result["predictions"], dtype=int)
        matrices[model_id] = np.asarray(result["confusion_matrix"], dtype=int)
        rows.append(
            {
                "model_id": model_id,
                "classifier": CLASSIFIER_DISPLAY_NAMES.get(classifier_id, classifier_id),
                "classifier_id": classifier_id,
                "feature_set": FEATURE_SET_DISPLAY_NAMES.get(feature_set_id, feature_set_id),
                "feature_set_id": feature_set_id,
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "validation_f1": artifact.get("validation_metrics", {}).get("f1"),
                "best_params": str(artifact.get("best_params", {})),
            }
        )

    return {
        "metrics": rows,
        "predictions": predictions,
        "confusion_matrices": matrices,
    }


def save_metrics(
    results: Union[Mapping[str, object], Sequence[Mapping[str, object]], pd.DataFrame],
    path: Union[str, Path],
) -> pd.DataFrame:
    """Write the ten-row comparison table to CSV and return it as a frame."""

    if isinstance(results, pd.DataFrame):
        metrics = results.copy()
    elif isinstance(results, Mapping):
        metrics = pd.DataFrame(results.get("metrics", []))
    else:
        metrics = pd.DataFrame(results)

    columns = [
        "model_id",
        "classifier",
        "classifier_id",
        "feature_set",
        "feature_set_id",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "validation_f1",
        "best_params",
    ]
    metrics = metrics.reindex(columns=columns)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination, index=False)
    return metrics


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def save_confusion_matrices(
    results: Mapping[str, object], directory: Union[str, Path]
) -> List[Path]:
    """Save one clearly labelled 2x2 CSV confusion matrix per evaluated model."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    matrices = results.get("confusion_matrices", {})
    if not isinstance(matrices, Mapping):
        raise ValueError("Evaluation results do not contain a confusion-matrix mapping.")

    for model_id, matrix in matrices.items():
        matrix_array = np.asarray(matrix, dtype=int)
        if matrix_array.shape != (2, 2):
            raise ValueError(f"Confusion matrix for '{model_id}' is not 2x2.")
        filename = f"{_safe_filename(str(model_id))}.csv"
        path = destination / filename
        pd.DataFrame(
            matrix_array,
            index=["actual_negative", "actual_positive"],
            columns=["predicted_negative", "predicted_positive"],
        ).to_csv(path, index_label="actual_label")
        saved.append(path)
    return saved


def save_error_analysis(
    reviews: Iterable[object],
    y_test: Iterable[int],
    results: Mapping[str, object],
    path: Union[str, Path],
) -> pd.DataFrame:
    """Record every incorrect held-out prediction for qualitative analysis."""

    review_list = ["" if review is None else str(review) for review in reviews]
    labels = np.asarray(list(y_test), dtype=int)
    if len(review_list) != len(labels):
        raise ValueError("reviews and y_test have different lengths.")

    metrics_by_id = {
        str(row["model_id"]): row
        for row in results.get("metrics", [])
        if isinstance(row, Mapping) and "model_id" in row
    }
    prediction_mapping = results.get("predictions", {})
    if not isinstance(prediction_mapping, Mapping):
        raise ValueError("Evaluation results do not contain prediction arrays.")

    errors: List[Dict[str, object]] = []
    for model_id, prediction_values in prediction_mapping.items():
        predictions = np.asarray(prediction_values, dtype=int)
        if len(predictions) != len(labels):
            raise ValueError(f"Prediction count for '{model_id}' does not match y_test.")
        metadata = metrics_by_id.get(str(model_id), {})
        for index in np.flatnonzero(predictions != labels):
            errors.append(
                {
                    "review_index": int(index),
                    "review": review_list[int(index)],
                    "true_label": _label_name(int(labels[index])),
                    "predicted_label": _label_name(int(predictions[index])),
                    "true_label_value": int(labels[index]),
                    "predicted_label_value": int(predictions[index]),
                    "classifier": metadata.get("classifier", ""),
                    "classifier_id": metadata.get("classifier_id", ""),
                    "feature_set": metadata.get("feature_set", ""),
                    "feature_set_id": metadata.get("feature_set_id", ""),
                    "model_id": str(model_id),
                }
            )

    error_frame = pd.DataFrame(
        errors,
        columns=[
            "review_index",
            "review",
            "true_label",
            "predicted_label",
            "true_label_value",
            "predicted_label_value",
            "classifier",
            "classifier_id",
            "feature_set",
            "feature_set_id",
            "model_id",
        ],
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    error_frame.to_csv(destination, index=False)
    return error_frame


def extract_feature_contributions(models: Mapping[str, object]) -> pd.DataFrame:
    """Return inspectable standardized linear-model coefficients when available.

    Positive coefficients push toward the positive class; negative coefficients
    push toward the negative class.  Coefficients are comparable within a model
    because every classifier pipeline scales its inputs first.
    """

    rows: List[Dict[str, object]] = []
    linguistic_names = set(ALL_FEATURE_NAMES)
    for artifact in iter_model_artifacts(models):
        estimator = artifact["estimator"]
        classifier = getattr(estimator, "named_steps", {}).get("classifier")
        coefficients = getattr(classifier, "coef_", None)
        if coefficients is None:
            continue
        coefficient_row = np.asarray(coefficients, dtype=float)
        if coefficient_row.ndim != 2 or coefficient_row.shape[0] != 1:
            continue
        feature_names = list(artifact.get("feature_names", []))
        if len(feature_names) != coefficient_row.shape[1]:
            continue
        for feature_name, coefficient in zip(feature_names, coefficient_row[0]):
            rows.append(
                {
                    "model_id": artifact["model_id"],
                    "classifier": CLASSIFIER_DISPLAY_NAMES.get(
                        str(artifact["classifier"]), str(artifact["classifier"])
                    ),
                    "feature_set": FEATURE_SET_DISPLAY_NAMES.get(
                        str(artifact["feature_set"]), str(artifact["feature_set"])
                    ),
                    "feature_name": feature_name,
                    "feature_group": "linguistic" if feature_name in linguistic_names else "doc2vec",
                    "coefficient": float(coefficient),
                    "absolute_coefficient": float(abs(coefficient)),
                }
            )
    contributions = pd.DataFrame(rows)
    if not contributions.empty:
        contributions = contributions.sort_values(
            ["model_id", "absolute_coefficient"], ascending=[True, False]
        ).reset_index(drop=True)
    return contributions


def save_feature_contributions(
    models: Mapping[str, object], path: Union[str, Path]
) -> pd.DataFrame:
    """Save model coefficients used to discuss useful linguistic features."""

    contributions = extract_feature_contributions(models)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contributions.to_csv(destination, index=False)
    return contributions

