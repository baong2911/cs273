"""A small Tkinter GUI for predicting hotel-review sentiment with saved models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

try:
    from .doc2vec_model import infer_review_vector, load_doc2vec
    from .feature_extraction import build_feature_matrices
    from .train_models import (
        CLASSIFIER_DISPLAY_NAMES,
        CLASSIFIER_ORDER,
        FEATURE_SET_DISPLAY_NAMES,
        FEATURE_SET_ORDER,
        iter_model_artifacts,
        load_models,
    )
except ImportError:  # pragma: no cover - supports ``python src/gui.py``
    from doc2vec_model import infer_review_vector, load_doc2vec
    from feature_extraction import build_feature_matrices
    from train_models import (
        CLASSIFIER_DISPLAY_NAMES,
        CLASSIFIER_ORDER,
        FEATURE_SET_DISPLAY_NAMES,
        FEATURE_SET_ORDER,
        iter_model_artifacts,
        load_models,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_PATH = PROJECT_ROOT / "models" / "classifiers.joblib"
DEFAULT_DOC2VEC_PATH = PROJECT_ROOT / "models" / "doc2vec.model"


def _model_mapping(model_bundle: Mapping[str, object]) -> Mapping[str, object]:
    models = model_bundle.get("models", model_bundle)
    if not isinstance(models, Mapping):
        raise ValueError("Model bundle does not contain a valid 'models' mapping.")
    return models


def available_model_choices(model_bundle: Mapping[str, object]) -> List[Tuple[str, str]]:
    """Return ``(model_id, user-facing label)`` choices in experiment order."""

    artifacts = {str(artifact["model_id"]): artifact for artifact in iter_model_artifacts(model_bundle)}
    choices: List[Tuple[str, str]] = []
    for classifier_id in CLASSIFIER_ORDER:
        for feature_set_id in FEATURE_SET_ORDER:
            model_id = f"{classifier_id}__{feature_set_id}"
            if model_id not in artifacts:
                continue
            label = (
                f"{CLASSIFIER_DISPLAY_NAMES.get(classifier_id, classifier_id)} — "
                f"{FEATURE_SET_DISPLAY_NAMES.get(feature_set_id, feature_set_id)}"
            )
            choices.append((model_id, label))
    # Keep malformed/custom bundles usable too, while normal saved bundles
    # always surface exactly the required ten choices above.
    for model_id in sorted(set(artifacts).difference({choice[0] for choice in choices})):
        artifact = artifacts[model_id]
        choices.append(
            (
                model_id,
                f"{artifact.get('classifier', '')} — {artifact.get('feature_set', '')}",
            )
        )
    return choices


def _representation_for_review(
    review: str, feature_set: str, doc2vec_model: Optional[object]
) -> np.ndarray:
    linguistic = build_feature_matrices([review])
    if feature_set == "basic_linguistic":
        return linguistic["basic"]
    if feature_set == "additional_linguistic":
        return linguistic["additional"]
    if feature_set == "all_linguistic":
        return linguistic["all_linguistic"]

    if feature_set in {"doc2vec", "combined"}:
        if doc2vec_model is None:
            raise ValueError("The selected model requires the saved Doc2Vec representation.")
        doc_vector = infer_review_vector(doc2vec_model, review).reshape(1, -1)
        if feature_set == "doc2vec":
            return doc_vector
        return np.hstack((linguistic["all_linguistic"], doc_vector))

    raise ValueError(f"Unsupported feature set '{feature_set}'.")


def predict_sentiment(
    review: str,
    model_id: str,
    model_bundle: Mapping[str, object],
    doc2vec_model: Optional[object] = None,
) -> str:
    """Predict ``Positive`` or ``Negative`` for one non-empty review.

    This pure helper is shared by the Tkinter callback and can also be used in
    a small script or test without starting a GUI window.
    """

    if not isinstance(review, str) or not review.strip():
        raise ValueError("Enter a hotel review before requesting a prediction.")
    models = _model_mapping(model_bundle)
    if model_id not in models:
        raise ValueError(f"Unknown saved model '{model_id}'.")
    artifact = models[model_id]
    if not isinstance(artifact, Mapping):
        raise ValueError(f"Saved model '{model_id}' is invalid.")
    if "estimator" not in artifact or "feature_set" not in artifact:
        raise ValueError(f"Saved model '{model_id}' is missing required metadata.")

    representation = _representation_for_review(
        review, str(artifact["feature_set"]), doc2vec_model
    )
    estimator = artifact["estimator"]
    expected_features = getattr(estimator, "n_features_in_", representation.shape[1])
    if representation.shape[1] != expected_features:
        raise ValueError(
            f"Feature dimension mismatch for '{model_id}': model expects "
            f"{expected_features}, but this representation has {representation.shape[1]}."
        )
    prediction = int(estimator.predict(representation)[0])
    return "Positive" if prediction == 1 else "Negative"


def launch_gui(
    models_path: Union[str, Path] = DEFAULT_MODELS_PATH,
    doc2vec_path: Union[str, Path] = DEFAULT_DOC2VEC_PATH,
) -> None:
    """Load saved artifacts and open the required hotel-review prediction UI."""

    # Import lazily so the core package remains usable on systems without a
    # graphical display (for example during command-line model evaluation).
    import tkinter as tk
    from tkinter import messagebox, ttk

    model_bundle = load_models(models_path)
    choices = available_model_choices(model_bundle)
    if not choices:
        raise ValueError("No trained classifiers were found in the saved model bundle.")
    requires_doc2vec = any(
        artifact["feature_set"] in {"doc2vec", "combined"}
        for artifact in iter_model_artifacts(model_bundle)
    )
    doc2vec_model = load_doc2vec(doc2vec_path) if requires_doc2vec else None

    root = tk.Tk()
    root.title("Hotel Review Sentiment Classifier")
    root.minsize(720, 470)

    container = ttk.Frame(root, padding=16)
    container.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)
    container.rowconfigure(1, weight=1)

    ttk.Label(container, text="Enter a hotel review:").grid(
        row=0, column=0, sticky="w", pady=(0, 6)
    )
    review_input = tk.Text(container, height=12, wrap="word")
    review_input.grid(row=1, column=0, sticky="nsew")

    ttk.Label(container, text="Trained model:").grid(
        row=2, column=0, sticky="w", pady=(14, 6)
    )
    labels_to_ids = {label: model_id for model_id, label in choices}
    selected_label = tk.StringVar(value=choices[0][1])
    selector = ttk.Combobox(
        container,
        textvariable=selected_label,
        values=list(labels_to_ids.keys()),
        state="readonly",
    )
    selector.grid(row=3, column=0, sticky="ew")

    result_text = tk.StringVar(value="Prediction: —")

    def on_predict() -> None:
        try:
            result = predict_sentiment(
                review_input.get("1.0", "end-1c"),
                labels_to_ids[selected_label.get()],
                model_bundle,
                doc2vec_model,
            )
            result_text.set(f"Prediction: {result}")
        except Exception as exc:  # Show user input/model file errors in the UI.
            result_text.set("Prediction: —")
            messagebox.showerror("Prediction unavailable", str(exc), parent=root)

    ttk.Button(container, text="Predict sentiment", command=on_predict).grid(
        row=4, column=0, sticky="w", pady=(14, 8)
    )
    ttk.Label(container, textvariable=result_text, font=("TkDefaultFont", 12, "bold")).grid(
        row=5, column=0, sticky="w"
    )
    root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the hotel-review sentiment GUI.")
    parser.add_argument("--models-path", type=Path, default=DEFAULT_MODELS_PATH)
    parser.add_argument("--doc2vec-path", type=Path, default=DEFAULT_DOC2VEC_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    launch_gui(args.models_path, args.doc2vec_path)

