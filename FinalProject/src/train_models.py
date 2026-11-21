"""Build the five required feature sets and train ten validation-selected models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from .feature_extraction import ALL_FEATURE_NAMES, BASIC_FEATURE_NAMES, ADDITIONAL_FEATURE_NAMES
except ImportError:  # pragma: no cover - only used for direct script execution
    from feature_extraction import ALL_FEATURE_NAMES, BASIC_FEATURE_NAMES, ADDITIONAL_FEATURE_NAMES


FEATURE_SET_ORDER: Tuple[str, ...] = (
    "basic_linguistic",
    "additional_linguistic",
    "all_linguistic",
    "doc2vec",
    "combined",
)
FEATURE_SET_DISPLAY_NAMES: Dict[str, str] = {
    "basic_linguistic": "10 basic linguistic features",
    "additional_linguistic": "5 additional linguistic features",
    "all_linguistic": "All 15 linguistic features",
    "doc2vec": "Doc2Vec vectors",
    "combined": "All 15 linguistic features + Doc2Vec",
}
CLASSIFIER_ORDER: Tuple[str, ...] = ("logistic_regression", "linear_svm")
CLASSIFIER_DISPLAY_NAMES: Dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
}
MODEL_BUNDLE_VERSION = 1


def _as_2d_array(matrix: object, name: str) -> np.ndarray:
    result = np.asarray(matrix, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix, got shape {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite feature values.")
    return result


def _doc_feature_names(width: int) -> List[str]:
    return [f"doc2vec_{index:03d}" for index in range(width)]


def build_feature_sets(
    train_linguistic: Mapping[str, object],
    validation_linguistic: Mapping[str, object],
    test_linguistic: Mapping[str, object],
    train_doc2vec: object,
    validation_doc2vec: object,
    test_doc2vec: object,
) -> Dict[str, Dict[str, object]]:
    """Create exactly the five proposal-required experimental feature sets.

    Each returned set contains matching ``train``, ``validation``, and ``test``
    arrays plus ordered ``feature_names``.  Linguistic matrices must come from
    ``feature_extraction.build_feature_matrices``.
    """

    linguistic_keys = ("basic", "additional", "all_linguistic")
    for split_name, matrices in (
        ("train", train_linguistic),
        ("validation", validation_linguistic),
        ("test", test_linguistic),
    ):
        missing = set(linguistic_keys).difference(matrices)
        if missing:
            raise ValueError(f"{split_name} linguistic matrices are missing keys: {sorted(missing)}")

    train_basic = _as_2d_array(train_linguistic["basic"], "train basic features")
    validation_basic = _as_2d_array(validation_linguistic["basic"], "validation basic features")
    test_basic = _as_2d_array(test_linguistic["basic"], "test basic features")
    train_additional = _as_2d_array(train_linguistic["additional"], "train additional features")
    validation_additional = _as_2d_array(
        validation_linguistic["additional"], "validation additional features"
    )
    test_additional = _as_2d_array(test_linguistic["additional"], "test additional features")
    train_all = _as_2d_array(train_linguistic["all_linguistic"], "train all linguistic features")
    validation_all = _as_2d_array(
        validation_linguistic["all_linguistic"], "validation all linguistic features"
    )
    test_all = _as_2d_array(test_linguistic["all_linguistic"], "test all linguistic features")
    train_doc = _as_2d_array(train_doc2vec, "train Doc2Vec features")
    validation_doc = _as_2d_array(validation_doc2vec, "validation Doc2Vec features")
    test_doc = _as_2d_array(test_doc2vec, "test Doc2Vec features")

    split_matrices = (
        ("train", train_basic, train_additional, train_all, train_doc),
        ("validation", validation_basic, validation_additional, validation_all, validation_doc),
        ("test", test_basic, test_additional, test_all, test_doc),
    )
    for split_name, basic, additional, all_linguistic, doc in split_matrices:
        if basic.shape[0] != additional.shape[0] or basic.shape[0] != all_linguistic.shape[0]:
            raise ValueError(f"{split_name} linguistic matrices have inconsistent row counts.")
        if basic.shape[0] != doc.shape[0]:
            raise ValueError(f"{split_name} linguistic and Doc2Vec matrices have inconsistent row counts.")
        if basic.shape[1] != len(BASIC_FEATURE_NAMES):
            raise ValueError("Basic linguistic features must have exactly 10 columns.")
        if additional.shape[1] != len(ADDITIONAL_FEATURE_NAMES):
            raise ValueError("Additional linguistic features must have exactly 5 columns.")
        if all_linguistic.shape[1] != len(ALL_FEATURE_NAMES):
            raise ValueError("All linguistic features must have exactly 15 columns.")

    doc_width = train_doc.shape[1]
    if validation_doc.shape[1] != doc_width or test_doc.shape[1] != doc_width:
        raise ValueError("Doc2Vec vector dimensions must match across all data splits.")

    return {
        "basic_linguistic": {
            "train": train_basic,
            "validation": validation_basic,
            "test": test_basic,
            "feature_names": list(BASIC_FEATURE_NAMES),
        },
        "additional_linguistic": {
            "train": train_additional,
            "validation": validation_additional,
            "test": test_additional,
            "feature_names": list(ADDITIONAL_FEATURE_NAMES),
        },
        "all_linguistic": {
            "train": train_all,
            "validation": validation_all,
            "test": test_all,
            "feature_names": list(ALL_FEATURE_NAMES),
        },
        "doc2vec": {
            "train": train_doc,
            "validation": validation_doc,
            "test": test_doc,
            "feature_names": _doc_feature_names(doc_width),
        },
        "combined": {
            "train": np.hstack((train_all, train_doc)),
            "validation": np.hstack((validation_all, validation_doc)),
            "test": np.hstack((test_all, test_doc)),
            "feature_names": list(ALL_FEATURE_NAMES) + _doc_feature_names(doc_width),
        },
    }


def _validate_training_inputs(
    X_train: object, y_train: Iterable[int], X_validation: object, y_validation: Iterable[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_matrix = _as_2d_array(X_train, "X_train")
    validation_matrix = _as_2d_array(X_validation, "X_validation")
    train_labels = np.asarray(list(y_train), dtype=int)
    validation_labels = np.asarray(list(y_validation), dtype=int)
    if train_matrix.shape[0] != len(train_labels):
        raise ValueError("X_train and y_train have different numbers of rows.")
    if validation_matrix.shape[0] != len(validation_labels):
        raise ValueError("X_validation and y_validation have different numbers of rows.")
    if train_matrix.shape[1] != validation_matrix.shape[1]:
        raise ValueError("Training and validation feature counts do not match.")
    if set(np.unique(train_labels)).difference({0, 1}) or set(np.unique(validation_labels)).difference({0, 1}):
        raise ValueError("Sentiment labels must be binary: 0 for negative and 1 for positive.")
    if len(np.unique(train_labels)) != 2:
        raise ValueError("Training data must contain both sentiment classes.")
    return train_matrix, train_labels, validation_matrix, validation_labels


def _validation_metrics(y_true: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def train_logistic_regression(
    X_train: object,
    y_train: Iterable[int],
    X_validation: object,
    y_validation: Iterable[int],
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, object]]:
    """Fit candidate logistic models and return the best validation-F1 model.

    The selected pipeline remains trained only on the 60% training partition.
    Validation data is used for selection, never fit into the saved pipeline.
    """

    train_matrix, train_labels, validation_matrix, validation_labels = _validate_training_inputs(
        X_train, y_train, X_validation, y_validation
    )
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("c_values must contain one or more positive values.")

    best_model = None
    best_info: Dict[str, object] = {}
    best_f1 = -1.0
    for c_value in c_values:
        candidate = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(c_value),
                        max_iter=2_000,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        candidate.fit(train_matrix, train_labels)
        metrics = _validation_metrics(validation_labels, candidate.predict(validation_matrix))
        # In a tie, preserve the earlier/smaller C candidate for a simpler,
        # deterministic choice.
        if metrics["f1"] > best_f1:
            best_model = candidate
            best_f1 = metrics["f1"]
            best_info = {"best_params": {"C": float(c_value)}, "validation_metrics": metrics}

    assert best_model is not None  # c_values validation above guarantees this.
    return best_model, best_info


def train_linear_svm(
    X_train: object,
    y_train: Iterable[int],
    X_validation: object,
    y_validation: Iterable[int],
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, object]]:
    """Fit candidate linear SVMs and return the best validation-F1 model."""

    train_matrix, train_labels, validation_matrix, validation_labels = _validate_training_inputs(
        X_train, y_train, X_validation, y_validation
    )
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("c_values must contain one or more positive values.")

    best_model = None
    best_info: Dict[str, object] = {}
    best_f1 = -1.0
    for c_value in c_values:
        candidate = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(C=float(c_value), max_iter=10_000, random_state=random_state),
                ),
            ]
        )
        candidate.fit(train_matrix, train_labels)
        metrics = _validation_metrics(validation_labels, candidate.predict(validation_matrix))
        if metrics["f1"] > best_f1:
            best_model = candidate
            best_f1 = metrics["f1"]
            best_info = {"best_params": {"C": float(c_value)}, "validation_metrics": metrics}

    assert best_model is not None
    return best_model, best_info


def train_all_models(
    feature_sets: Mapping[str, Mapping[str, object]],
    y_train: Iterable[int],
    y_validation: Iterable[int],
    random_state: int = 42,
) -> Dict[str, object]:
    """Train both required classifiers on all five required feature setups."""

    missing_feature_sets = set(FEATURE_SET_ORDER).difference(feature_sets)
    if missing_feature_sets:
        raise ValueError(f"Missing required feature sets: {sorted(missing_feature_sets)}")

    train_labels = np.asarray(list(y_train), dtype=int)
    validation_labels = np.asarray(list(y_validation), dtype=int)
    trained: Dict[str, Dict[str, object]] = {}
    for feature_set in FEATURE_SET_ORDER:
        matrices = feature_sets[feature_set]
        for required_key in ("train", "validation", "feature_names"):
            if required_key not in matrices:
                raise ValueError(f"Feature set '{feature_set}' is missing '{required_key}'.")
        feature_names = list(matrices["feature_names"])
        for classifier_name in CLASSIFIER_ORDER:
            if classifier_name == "logistic_regression":
                estimator, selection = train_logistic_regression(
                    matrices["train"],
                    train_labels,
                    matrices["validation"],
                    validation_labels,
                    random_state=random_state,
                )
            else:
                estimator, selection = train_linear_svm(
                    matrices["train"],
                    train_labels,
                    matrices["validation"],
                    validation_labels,
                    random_state=random_state,
                )

            model_id = f"{classifier_name}__{feature_set}"
            trained[model_id] = {
                "model_id": model_id,
                "classifier": classifier_name,
                "feature_set": feature_set,
                "estimator": estimator,
                "feature_names": feature_names,
                **selection,
            }

    return {
        "bundle_version": MODEL_BUNDLE_VERSION,
        "models": trained,
        "feature_set_display_names": dict(FEATURE_SET_DISPLAY_NAMES),
        "classifier_display_names": dict(CLASSIFIER_DISPLAY_NAMES),
    }


def iter_model_artifacts(models: Mapping[str, object]) -> Iterator[Dict[str, object]]:
    """Yield saved model artifacts from a model bundle or legacy flat mapping."""

    model_mapping = models.get("models", models)
    if not isinstance(model_mapping, Mapping):
        raise ValueError("Model bundle does not contain a valid 'models' mapping.")
    for model_id, artifact in model_mapping.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Model artifact '{model_id}' is not a mapping.")
        if "estimator" not in artifact or "feature_set" not in artifact or "classifier" not in artifact:
            raise ValueError(f"Model artifact '{model_id}' is missing required metadata.")
        yield dict(artifact)


def save_models(models: Mapping[str, object], path: Union[str, Path]) -> None:
    """Persist all ten classifier pipelines and their feature metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(models), destination)


def load_models(path: Union[str, Path]) -> Dict[str, object]:
    """Load a classifier bundle saved by ``save_models``."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Classifier bundle not found at {source}. Run `python main.py` before launching the GUI."
        )
    models = joblib.load(source)
    if not isinstance(models, Mapping):
        raise ValueError("Saved classifiers file does not contain a model bundle.")
    # Validate eagerly so GUI errors are understandable before opening a window.
    list(iter_model_artifacts(models))
    return dict(models)

