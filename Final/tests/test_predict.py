import numpy as np
import pytest

from src.config import DOC2VEC_VECTOR_SIZE
from src.feature_builder import build_feature_sets
from src.linguistic_features import extract_linguistic_matrix
from src.predict import predict_review
from src.train import fit_model


class DummyDoc2Vec:
    """Small inference double that follows the Doc2Vec API used by predict."""

    vector_size = DOC2VEC_VECTOR_SIZE
    epochs = 2

    def infer_vector(self, tokens, epochs=2):
        value = float(sum(len(token) for token in tokens))
        return np.full(self.vector_size, value, dtype=np.float32)


def test_linguistic_prediction_and_blank_input():
    texts = ["The room was clean and wonderful.", "The room was dirty and terrible."]
    linguistic = extract_linguistic_matrix(texts)
    model = fit_model(linguistic.to_numpy(), np.array([1, 0]), "logistic", C=1.0)

    result = predict_review("The hotel was wonderful and clean!", model=model, feature_set="all15")
    assert result["sentiment"] in {"positive", "negative"}
    assert result["label"] in {0, 1}
    with pytest.raises(ValueError):
        predict_review("   ", model=model, feature_set="all15")


def test_doc2vec_routing_requires_artifact_and_can_infer_with_one():
    texts = ["Wonderful clean hotel.", "Terrible dirty hotel."]
    linguistic = extract_linguistic_matrix(texts)
    dummy_vectors = np.vstack(
        [np.zeros(DOC2VEC_VECTOR_SIZE), np.ones(DOC2VEC_VECTOR_SIZE)]
    )
    training_sets = build_feature_sets(texts, linguistic, dummy_vectors)
    model = fit_model(training_sets["doc2vec"], np.array([1, 0]), "svm", C=1.0)

    with pytest.raises((FileNotFoundError, ImportError)):
        predict_review("A new review", model=model, feature_set="doc2vec")

    result = predict_review(
        "A new review",
        model=model,
        feature_set="doc2vec",
        doc2vec_model=DummyDoc2Vec(),
    )
    assert result["sentiment"] in {"positive", "negative"}
    assert result["score_type"] == "decision_function"
