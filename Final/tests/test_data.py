import pandas as pd

from src.data import clean_and_label, create_balanced_sample, split_dataset


def test_cleaning_labels_and_duplicate_blank_rules():
    raw = pd.DataFrame(
        {
            "Review": [
                "  Terrible room  ",
                "terrible   room",  # normalized duplicate of the first row
                "Average stay",
                "Wonderful staff",
                "Awful service",
                "Excellent location",
                "",
                None,
            ],
            "Rating": [1, 2, 3, 4, 2, 5, 4, 5],
        }
    )
    cleaned = clean_and_label(raw)

    assert list(cleaned["review"]) == [
        "Terrible room",
        "Wonderful staff",
        "Awful service",
        "Excellent location",
    ]
    assert list(cleaned["label"]) == [0, 1, 0, 1]
    assert set(cleaned["rating"]) == {1, 2, 4, 5}


def test_balanced_sample_and_stratified_split_sizes():
    raw = pd.DataFrame(
        {
            "Review": [f"review {index}" for index in range(20)],
            "Rating": [1] * 10 + [5] * 10,
        }
    )
    cleaned = clean_and_label(raw)
    balanced = create_balanced_sample(cleaned, per_class=10, seed=7)
    train, validation, test = split_dataset(balanced, seed=7)

    assert len(balanced) == 20
    assert balanced["label"].value_counts().to_dict() == {0: 10, 1: 10}
    assert [len(frame) for frame in (train, validation, test)] == [12, 4, 4]
    for frame in (train, validation, test):
        assert frame["label"].value_counts().to_dict() == {
            0: len(frame) // 2,
            1: len(frame) // 2,
        }
