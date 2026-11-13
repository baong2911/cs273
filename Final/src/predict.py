"""Inference API shared by the GUI and programmatic callers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    CLASSIFIER_DISPLAY_NAMES,
    CLASSIFIER_NAMES,
    DOC2VEC_MODEL_PATH,
    FEATURE_SET_NAMES,
    MODELS_DIR,
)
from .doc2vec_features import infer_vectors, load_doc2vec
from .feature_builder import build_feature_sets
from .linguistic_features import extract_linguistic_matrix
from .train import load_model, normalize_classifier_name


def model_choices(models_dir: Path | str = MODELS_DIR) -> list[dict[str, Any]]:
    """Return the ten stable GUI model choices and their artifact paths."""

    models_dir = Path(models_dir)
    choices = []
    for feature_set in FEATURE_SET_NAMES:
        for classifier in CLASSIFIER_NAMES:
            path = models_dir / model_path(classifier, feature_set).name
            choices.append(
                {
                    "id": f"{classifier}_{feature_set}",
                    "display_name": f"{CLASSIFIER_DISPLAY_NAMES[classifier]} — {feature_set}",
                    "classifier": classifier,
                    "feature_set": feature_set,
                    "path": path,
                }
            )
    return choices


def _validate_review(review: str) -> str:
    if review is None or not str(review).strip():
        raise ValueError("Please enter a non-blank review.")
    return str(review)


def predict_review(
    review: str,
    model=None,
    feature_set: str | None = None,
    classifier: str | None = None,
    model_path_value: Path | str | None = None,
    model_path: Path | str | None = None,
    doc2vec_model=None,
    doc2vec_path: Path | str = DOC2VEC_MODEL_PATH,
) -> dict[str, Any]:
    """Classify one review using a saved or already-loaded model.

    ``model`` and a model-path argument are alternatives. ``model_path`` is the
    public spelling; ``model_path_value`` is retained as an explicit alias for
    callers that prefer a non-shadowing parameter name. ``feature_set`` is
    required when passing a model object; the filename is sufficient when using
    a canonical model path.
    """

    review = _validate_review(review)
    if model_path is not None:
        if model_path_value is not None:
            raise ValueError("Use only one of model_path and model_path_value")
        model_path_value = model_path
    if model is None:
        if model_path_value is None:
            raise ValueError("Provide either model or model_path_value")
        model_path_obj = Path(model_path_value)
        model = load_model(model_path_obj)
        if feature_set is None or classifier is None:
            stem_parts = model_path_obj.stem.split("_", maxsplit=1)
            if len(stem_parts) == 2:
                classifier = classifier or normalize_classifier_name(stem_parts[0])
                feature_set = feature_set or stem_parts[1]
    if feature_set not in FEATURE_SET_NAMES:
        raise ValueError(f"Unknown or missing feature_set: {feature_set}")
    if classifier is not None:
        classifier = normalize_classifier_name(classifier)

    linguistic = extract_linguistic_matrix([review])
    needs_doc2vec = feature_set in {"doc2vec", "all15_doc2vec"}
    vectors = None
    if needs_doc2vec:
        if doc2vec_model is None:
            doc2vec_model = load_doc2vec(doc2vec_path)
        vectors = infer_vectors(doc2vec_model, [review])
    feature_sets = build_feature_sets(
        [review], linguistic_features=linguistic, doc2vec_vectors=vectors
    )
    if feature_set not in feature_sets:
        raise ValueError(f"Feature set '{feature_set}' could not be assembled")

    matrix = feature_sets[feature_set]
    prediction = int(model.predict(matrix)[0])
    result: dict[str, Any] = {
        "label": prediction,
        "sentiment": "positive" if prediction == 1 else "negative",
        "feature_set": feature_set,
    }
    if classifier:
        result["classifier"] = classifier
    if hasattr(model, "predict_proba"):
        result["positive_probability"] = float(model.predict_proba(matrix)[0, 1])
        result["score_type"] = "probability"
    elif hasattr(model, "decision_function"):
        result["decision_score"] = float(model.decision_function(matrix)[0])
        result["score_type"] = "decision_function"
    return result


def predict_from_choice(
    review: str,
    choice: dict[str, Any],
    doc2vec_path: Path | str = DOC2VEC_MODEL_PATH,
) -> dict[str, Any]:
    """Convenience wrapper for a choice returned by :func:`model_choices`."""

    return predict_review(
        review,
        feature_set=choice["feature_set"],
        classifier=choice["classifier"],
        model_path_value=choice["path"],
        doc2vec_path=doc2vec_path,
    )
