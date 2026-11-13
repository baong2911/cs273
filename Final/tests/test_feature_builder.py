import numpy as np

from src.config import DOC2VEC_VECTOR_SIZE
from src.feature_builder import build_feature_sets
from src.linguistic_features import extract_linguistic_matrix


def test_canonical_feature_set_dimensions_and_row_order():
    texts = ["The clean room was great.", "The service was terrible!"]
    linguistic = extract_linguistic_matrix(texts)
    doc2vec = np.arange(2 * DOC2VEC_VECTOR_SIZE, dtype=float).reshape(
        2, DOC2VEC_VECTOR_SIZE
    )
    feature_sets = build_feature_sets(texts, linguistic, doc2vec)

    assert list(feature_sets) == ["basic10", "extra5", "all15", "doc2vec", "all15_doc2vec"]
    assert feature_sets["basic10"].shape == (2, 10)
    assert feature_sets["extra5"].shape == (2, 5)
    assert feature_sets["all15"].shape == (2, 15)
    assert feature_sets["doc2vec"].shape == (2, DOC2VEC_VECTOR_SIZE)
    assert feature_sets["all15_doc2vec"].shape == (2, 15 + DOC2VEC_VECTOR_SIZE)
    np.testing.assert_array_equal(feature_sets["doc2vec"][0], doc2vec[0])
    np.testing.assert_array_equal(feature_sets["doc2vec"][1], doc2vec[1])
