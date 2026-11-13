"""Project-wide constants and paths.

This module deliberately contains configuration only. Data processing and
modeling logic belongs in the other ``src`` modules.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "tripadvisor_hotel_reviews.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOC2VEC_DIR = ARTIFACTS_DIR / "doc2vec"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"

BALANCED_DATA_PATH = PROCESSED_DATA_DIR / "balanced_6000.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
DOC2VEC_MODEL_PATH = DOC2VEC_DIR / "doc2vec.model"

RANDOM_SEED = 273

RAW_REVIEW_COLUMN = "Review"
RAW_RATING_COLUMN = "Rating"
REVIEW_COLUMN = "review"
RATING_COLUMN = "rating"
LABEL_COLUMN = "label"
REVIEW_ID_COLUMN = "review_id"

LABEL_MAPPING = {1: 0, 2: 0, 4: 1, 5: 1}
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
SAMPLE_PER_CLASS = 3000
TOTAL_SAMPLE_SIZE = SAMPLE_PER_CLASS * 2
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20

BASIC_FEATURES = (
    "avg_word_length",
    "avg_sentence_length",
    "word_count",
    "verb_count",
    "adjective_count",
    "passive_voice_count",
    "sentence_count",
    "content_word_diversity",
    "typo_count",
    "typo_ratio",
)
EXTRA_FEATURES = (
    "negation_count",
    "first_person_pronoun_ratio",
    "adverb_count",
    "exclamation_count",
    "sentiment_lexicon_score",
)
ALL_LINGUISTIC_FEATURES = BASIC_FEATURES + EXTRA_FEATURES

FEATURE_SET_NAMES = (
    "basic10",
    "extra5",
    "all15",
    "doc2vec",
    "all15_doc2vec",
)
LINGUISTIC_FEATURE_SET_NAMES = ("basic10", "extra5", "all15")

DOC2VEC_VECTOR_SIZE = 100
DOC2VEC_WINDOW = 5
DOC2VEC_MIN_COUNT = 2
DOC2VEC_WORKERS = 1
DOC2VEC_EPOCHS = 40

CLASSIFIER_NAMES = ("logistic", "svm")
CLASSIFIER_DISPLAY_NAMES = {
    "logistic": "Logistic Regression",
    "svm": "Linear SVM",
}
CLASSIFIER_C_VALUES = (0.1, 1.0, 10.0)

VALIDATION_RESULTS_PATH = RESULTS_DIR / "validation_results.csv"
TEST_RESULTS_PATH = RESULTS_DIR / "test_results.csv"
ERRORS_PATH = RESULTS_DIR / "errors.csv"
LINGUISTIC_COEFFICIENTS_PATH = RESULTS_DIR / "linguistic_coefficients.csv"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


def model_path(classifier: str, feature_set: str) -> Path:
    """Return the canonical path for a final saved model."""

    if classifier not in CLASSIFIER_NAMES:
        raise ValueError(f"Unknown classifier: {classifier}")
    if feature_set not in FEATURE_SET_NAMES:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return MODELS_DIR / f"{classifier}_{feature_set}.joblib"
