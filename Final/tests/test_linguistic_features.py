import numpy as np

from src.linguistic_features import extract_linguistic_features, extract_linguistic_matrix


def test_predictable_punctuation_negation_and_sentence_features():
    features = extract_linguistic_features("I am very happy! I do not hate it.")

    assert features["exclamation_count"] == 1
    assert features["negation_count"] == 1
    assert features["first_person_pronoun_ratio"] == 2 / 9
    assert features["sentence_count"] == 2
    assert features["word_count"] == 9
    assert features["adverb_count"] >= 1
    assert features["verb_count"] >= 2


def test_passive_voice_heuristic_and_finite_empty_features():
    passive = extract_linguistic_features("The room was cleaned by staff.")
    empty = extract_linguistic_features("")
    matrix = extract_linguistic_matrix(["", "A great stay!"])

    assert passive["passive_voice_count"] >= 1
    assert all(value == 0 for value in empty.values())
    assert np.isfinite(matrix.to_numpy()).all()
    assert (matrix.iloc[0] == 0).all()
