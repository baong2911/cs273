"""Run the complete train/validation/test experiment from prepared CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import (
    CLASSIFIER_C_VALUES,
    CLASSIFIER_DISPLAY_NAMES,
    CLASSIFIER_NAMES,
    DOC2VEC_EPOCHS,
    DOC2VEC_MIN_COUNT,
    DOC2VEC_MODEL_PATH,
    DOC2VEC_VECTOR_SIZE,
    DOC2VEC_WINDOW,
    DOC2VEC_WORKERS,
    ERRORS_PATH,
    FEATURE_SET_NAMES,
    LINGUISTIC_COEFFICIENTS_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_DATA_PATH,
    TEST_RESULTS_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    VALIDATION_RESULTS_PATH,
    model_path,
)
from .doc2vec_features import (
    DEFAULT_DOC2VEC_CONFIG,
    infer_vectors,
    save_doc2vec,
    train_doc2vec,
    training_vectors,
)
from .evaluate import (
    build_error_table,
    compute_metrics,
    extract_linguistic_coefficients,
    save_confusion_matrix,
)
from .feature_builder import build_feature_sets, save_feature_sets
from .linguistic_features import extract_linguistic_matrix
from .train import fit_model, save_model


RESULT_COLUMNS = [
    "feature_set",
    "classifier",
    "C",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "random_seed",
    "doc2vec_vector_size",
    "doc2vec_window",
    "doc2vec_min_count",
    "doc2vec_workers",
    "doc2vec_epochs",
]


def _load_processed_splits(processed_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    paths = {
        "train": processed_dir / TRAIN_DATA_PATH.name,
        "validation": processed_dir / VALIDATION_DATA_PATH.name,
        "test": processed_dir / TEST_DATA_PATH.name,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared split(s) not found: " + ", ".join(missing) + ". Run `python -m src.data` first."
        )
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    required = {"review_id", "review", "rating", "label"}
    for name, frame in frames.items():
        if not required.issubset(frame.columns):
            raise ValueError(f"{name}.csv must contain columns {sorted(required)}")
        if frame["review"].isna().any() or frame["label"].isna().any():
            raise ValueError(f"{name}.csv contains missing review text or label")
    return frames["train"], frames["validation"], frames["test"]


def _configuration_fields(seed: int, d2v_config: Mapping[str, int]) -> dict[str, int]:
    return {
        "random_seed": int(seed),
        "doc2vec_vector_size": int(d2v_config["vector_size"]),
        "doc2vec_window": int(d2v_config["window"]),
        "doc2vec_min_count": int(d2v_config["min_count"]),
        "doc2vec_workers": int(d2v_config["workers"]),
        "doc2vec_epochs": int(d2v_config["epochs"]),
    }


def _make_result_row(
    feature_set: str,
    classifier: str,
    C: float,
    metrics: Mapping[str, float],
    seed: int,
    d2v_config: Mapping[str, int],
) -> dict:
    row = {
        "feature_set": feature_set,
        "classifier": classifier,
        "C": float(C),
        **metrics,
        **_configuration_fields(seed, d2v_config),
    }
    return {column: row[column] for column in RESULT_COLUMNS}


def _select_best(validation_results: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Select C by F1, then accuracy, then distance from the default C=1."""

    selected: dict[tuple[str, str], float] = {}
    for feature_set in FEATURE_SET_NAMES:
        for classifier in CLASSIFIER_NAMES:
            group = validation_results[
                (validation_results["feature_set"] == feature_set)
                & (validation_results["classifier"] == classifier)
            ].copy()
            if group.empty:
                raise ValueError(f"No validation results for {classifier}/{feature_set}")
            group["_distance_from_default"] = (group["C"] - 1.0).abs()
            group = group.sort_values(
                by=["f1", "accuracy", "_distance_from_default", "C"],
                ascending=[False, False, True, True],
                kind="mergesort",
            )
            selected[(feature_set, classifier)] = float(group.iloc[0]["C"])
    return selected


def _build_selection_representations(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    d2v_config: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract model-selection representations without test leakage."""

    train_texts = train["review"].astype(str).tolist()
    validation_texts = validation["review"].astype(str).tolist()
    test_texts = test["review"].astype(str).tolist()

    train_linguistic = extract_linguistic_matrix(train_texts)
    validation_linguistic = extract_linguistic_matrix(validation_texts)
    test_linguistic = extract_linguistic_matrix(test_texts)

    selection_doc2vec = train_doc2vec(train_texts, config=d2v_config, seed=seed)
    train_vectors = training_vectors(selection_doc2vec, len(train_texts))
    validation_vectors = infer_vectors(selection_doc2vec, validation_texts, seed=seed)
    test_vectors = infer_vectors(selection_doc2vec, test_texts, seed=seed)

    train_sets = build_feature_sets(
        train_texts, linguistic_features=train_linguistic, doc2vec_vectors=train_vectors
    )
    validation_sets = build_feature_sets(
        validation_texts,
        linguistic_features=validation_linguistic,
        doc2vec_vectors=validation_vectors,
    )
    test_sets = build_feature_sets(
        test_texts, linguistic_features=test_linguistic, doc2vec_vectors=test_vectors
    )
    return train_sets, validation_sets, test_sets


def _validation_search(
    train_sets: Mapping[str, np.ndarray],
    validation_sets: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    d2v_config: Mapping[str, int],
) -> pd.DataFrame:
    rows = []
    for feature_set in FEATURE_SET_NAMES:
        for classifier in CLASSIFIER_NAMES:
            for C in CLASSIFIER_C_VALUES:
                model = fit_model(
                    train_sets[feature_set], y_train, classifier_name=classifier, C=C
                )
                predictions = model.predict(validation_sets[feature_set])
                rows.append(
                    _make_result_row(
                        feature_set,
                        classifier,
                        C,
                        compute_metrics(y_validation, predictions),
                        seed,
                        d2v_config,
                    )
                )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _build_final_representations(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    d2v_config: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], object]:
    """Retrain Doc2Vec on train+validation after validation choices are frozen."""

    combined = pd.concat([train, validation], ignore_index=True)
    combined_texts = combined["review"].astype(str).tolist()
    test_texts = test["review"].astype(str).tolist()
    combined_linguistic = extract_linguistic_matrix(combined_texts)
    test_linguistic = extract_linguistic_matrix(test_texts)

    final_doc2vec = train_doc2vec(combined_texts, config=d2v_config, seed=seed)
    combined_vectors = training_vectors(final_doc2vec, len(combined_texts))
    test_vectors = infer_vectors(final_doc2vec, test_texts, seed=seed)
    combined_sets = build_feature_sets(
        combined_texts,
        linguistic_features=combined_linguistic,
        doc2vec_vectors=combined_vectors,
    )
    test_sets = build_feature_sets(
        test_texts, linguistic_features=test_linguistic, doc2vec_vectors=test_vectors
    )
    return combined_sets, test_sets, final_doc2vec


def run_experiments(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    artifacts_dir: Path | str | None = None,
    seed: int = RANDOM_SEED,
    d2v_config: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run validation selection and the ten final held-out test evaluations."""

    processed_dir = Path(processed_dir)
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir = Path(artifacts_dir)
    features_dir = artifacts_dir / "features"
    models_dir = artifacts_dir / "models"
    results_dir = artifacts_dir / "results"
    matrices_dir = results_dir / "confusion_matrices"
    doc2vec_dir = artifacts_dir / "doc2vec"
    d2v_params = dict(DEFAULT_DOC2VEC_CONFIG)
    if d2v_config:
        d2v_params.update(d2v_config)

    train, validation, test = _load_processed_splits(processed_dir)
    train_sets, validation_sets, test_selection_sets = _build_selection_representations(
        train, validation, test, d2v_params, seed
    )
    for split_name, feature_sets in (
        ("train", train_sets),
        ("validation", validation_sets),
        ("test", test_selection_sets),
    ):
        save_feature_sets(feature_sets, split_name, output_dir=features_dir)

    y_train = train["label"].to_numpy(dtype=int)
    y_validation = validation["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)
    validation_results = _validation_search(
        train_sets,
        validation_sets,
        y_train,
        y_validation,
        seed,
        d2v_params,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(results_dir / VALIDATION_RESULTS_PATH.name, index=False)
    selected_C = _select_best(validation_results)

    final_train_sets, final_test_sets, final_doc2vec = _build_final_representations(
        train, validation, test, d2v_params, seed
    )
    save_feature_sets(final_train_sets, "train_validation", output_dir=features_dir)
    save_feature_sets(final_test_sets, "test_final", output_dir=features_dir)
    save_doc2vec(final_doc2vec, doc2vec_dir / DOC2VEC_MODEL_PATH.name)

    test_rows = []
    error_frames = []
    coefficient_frames = []
    metadata = {
        "random_seed": int(seed),
        "doc2vec": {key: int(value) for key, value in d2v_params.items()},
        "selected_C": {f"{feature_set}/{classifier}": C for (feature_set, classifier), C in selected_C.items()},
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    for feature_set in FEATURE_SET_NAMES:
        for classifier in CLASSIFIER_NAMES:
            C = selected_C[(feature_set, classifier)]
            model = fit_model(
                final_train_sets[feature_set],
                np.concatenate([y_train, y_validation]),
                classifier_name=classifier,
                C=C,
            )
            predictions = model.predict(final_test_sets[feature_set])
            test_rows.append(
                _make_result_row(
                    feature_set,
                    classifier,
                    C,
                    compute_metrics(y_test, predictions),
                    seed,
                    d2v_params,
                )
            )
            save_model(model, models_dir / model_path(classifier, feature_set).name)
            save_confusion_matrix(
                y_test,
                predictions,
                matrices_dir / f"{classifier}_{feature_set}.png",
                title=f"{CLASSIFIER_DISPLAY_NAMES[classifier]} — {feature_set}",
            )
            error_frames.append(
                build_error_table(test, y_test, predictions, feature_set, classifier)
            )
            coefficient_frames.append(
                extract_linguistic_coefficients(model, feature_set, classifier)
            )

    test_results = pd.DataFrame(test_rows, columns=RESULT_COLUMNS)
    test_results.to_csv(results_dir / TEST_RESULTS_PATH.name, index=False)
    errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    errors.to_csv(results_dir / ERRORS_PATH.name, index=False)
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame()
    )
    coefficients.to_csv(results_dir / LINGUISTIC_COEFFICIENTS_PATH.name, index=False)
    with (models_dir / MODEL_METADATA_PATH.name).open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print("Final test results ranked by F1:")
    ranked = test_results.sort_values(
        by=["f1", "accuracy"], ascending=[False, False], kind="mergesort"
    )
    for row in ranked.itertuples(index=False):
        print(
            f"  {row.feature_set:16s} {row.classifier:8s} "
            f"C={row.C:g} accuracy={row.accuracy:.3f} f1={row.f1:.3f}"
        )
    return validation_results, test_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--doc2vec-epochs", type=int, default=DOC2VEC_EPOCHS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = {
        "vector_size": DOC2VEC_VECTOR_SIZE,
        "window": DOC2VEC_WINDOW,
        "min_count": DOC2VEC_MIN_COUNT,
        "workers": DOC2VEC_WORKERS,
        "epochs": args.doc2vec_epochs,
    }
    run_experiments(
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
        d2v_config=config,
    )


if __name__ == "__main__":
    main()
