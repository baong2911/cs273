"""Command-line workflow for the Hotel Review Sentiment Classification project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

from src.data_preparation import (
    DEFAULT_RANDOM_STATE,
    create_balanced_sample,
    create_sentiment_labels,
    load_and_clean_data,
    split_data,
    split_summary,
    tokenize_reviews,
)
from src.doc2vec_model import (
    DEFAULT_EPOCHS,
    DEFAULT_VECTOR_SIZE,
    get_doc2vec_vectors,
    save_doc2vec,
    train_doc2vec,
)
from src.evaluate_models import (
    evaluate_all_models,
    save_confusion_matrices,
    save_error_analysis,
    save_feature_contributions,
    save_metrics,
)
from src.feature_extraction import build_feature_matrices
from src.nlp_utils import ensure_nltk_resources
from src.train_models import build_feature_sets, save_models, train_all_models


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the five required hotel-review feature setups "
            "with Logistic Regression and linear SVM."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "tripadvisor_hotel_reviews.csv",
        help="Path to the TripAdvisor review CSV.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory for doc2vec.model and classifiers.joblib.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for metrics, confusion matrices, and error analysis.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for sampling, splitting, and model training.",
    )
    parser.add_argument(
        "--doc2vec-vector-size",
        type=int,
        default=DEFAULT_VECTOR_SIZE,
        help="Doc2Vec embedding dimension (default: %(default)s).",
    )
    parser.add_argument(
        "--doc2vec-epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Doc2Vec training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--doc2vec-min-count",
        type=int,
        default=2,
        help="Minimum Doc2Vec token frequency (default: %(default)s).",
    )
    parser.add_argument(
        "--doc2vec-workers",
        type=int,
        default=1,
        help="Doc2Vec workers; 1 gives the most reproducible experiment.",
    )
    parser.add_argument(
        "--download-nltk",
        action="store_true",
        help="Download missing NLTK tokenizer, POS-tagger, and VADER data first.",
    )
    return parser.parse_args()


def run_workflow(args: argparse.Namespace) -> Dict[str, object]:
    """Run the documented data, representation, training, and evaluation flow."""

    nltk_status = ensure_nltk_resources(download=args.download_nltk)
    missing_resources = [name for name, installed in nltk_status.items() if not installed]
    if missing_resources:
        LOGGER.warning(
            "Some NLTK data is unavailable (%s). The code has limited fallbacks, "
            "but run again with --download-nltk for the intended tokenizer, POS, "
            "and VADER features.",
            ", ".join(missing_resources),
        )

    LOGGER.info("Loading and cleaning %s", args.data_path)
    cleaned = load_and_clean_data(args.data_path)
    labeled = create_sentiment_labels(cleaned)
    balanced = create_balanced_sample(labeled, random_state=args.random_state)
    tokenized = tokenize_reviews(balanced)
    train, validation, test = split_data(tokenized, random_state=args.random_state)
    LOGGER.info("Stratified split: %s", split_summary(train, validation, test))

    # All three linguistic matrices are calculated once per split and reused by
    # the required basic/additional/all/combined feature experiments.
    LOGGER.info("Extracting the 15 linguistic features")
    train_linguistic = build_feature_matrices(train["review"])
    validation_linguistic = build_feature_matrices(validation["review"])
    test_linguistic = build_feature_matrices(test["review"])

    LOGGER.info("Training Doc2Vec on %d training reviews only", len(train))
    doc2vec = train_doc2vec(
        train["review"],
        vector_size=args.doc2vec_vector_size,
        epochs=args.doc2vec_epochs,
        random_state=args.random_state,
        min_count=args.doc2vec_min_count,
        workers=args.doc2vec_workers,
    )
    args.models_dir.mkdir(parents=True, exist_ok=True)
    doc2vec_path = args.models_dir / "doc2vec.model"
    save_doc2vec(doc2vec, doc2vec_path)

    train_doc2vec_vectors = get_doc2vec_vectors(doc2vec, train["review"], training=True)
    validation_doc2vec_vectors = get_doc2vec_vectors(doc2vec, validation["review"])
    test_doc2vec_vectors = get_doc2vec_vectors(doc2vec, test["review"])
    feature_sets = build_feature_sets(
        train_linguistic,
        validation_linguistic,
        test_linguistic,
        train_doc2vec_vectors,
        validation_doc2vec_vectors,
        test_doc2vec_vectors,
    )

    LOGGER.info("Training 10 Logistic Regression / linear SVM combinations")
    models = train_all_models(
        feature_sets,
        train["sentiment"],
        validation["sentiment"],
        random_state=args.random_state,
    )
    classifiers_path = args.models_dir / "classifiers.joblib"
    save_models(models, classifiers_path)

    LOGGER.info("Evaluating selected models once on the held-out test split")
    evaluation = evaluate_all_models(models, feature_sets, test["sentiment"])
    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics = save_metrics(evaluation, args.results_dir / "metrics.csv")
    matrix_paths = save_confusion_matrices(evaluation, args.results_dir / "confusion_matrices")
    errors = save_error_analysis(
        test["review"], test["sentiment"], evaluation, args.results_dir / "error_analysis.csv"
    )
    contributions = save_feature_contributions(
        models, args.results_dir / "feature_contributions.csv"
    )

    LOGGER.info("Saved %s", classifiers_path)
    LOGGER.info("Saved %s", doc2vec_path)
    LOGGER.info("Saved %s", args.results_dir / "metrics.csv")
    LOGGER.info("Saved %d confusion matrices and %d error-analysis rows", len(matrix_paths), len(errors))
    LOGGER.info("Saved %d inspectable linear-feature coefficients", len(contributions))
    return {
        "models": models,
        "metrics": metrics,
        "evaluation": evaluation,
        "train": train,
        "validation": validation,
        "test": test,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    workflow = run_workflow(args)
    metrics = workflow["metrics"]
    # A compact terminal table is useful while the full CSV remains the formal
    # comparison artifact for the report.
    print("\nFinal held-out test metrics:")
    print(metrics[["classifier", "feature_set", "accuracy", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()

