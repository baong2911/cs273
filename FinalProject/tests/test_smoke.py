"""Small offline checks for the project modules without the Kaggle dataset."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preparation import create_sentiment_labels, split_data
from src.doc2vec_model import (
    get_doc2vec_vectors,
    load_doc2vec,
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
from src.feature_extraction import build_feature_matrices, extract_all_features
from src.gui import predict_sentiment
from src.train_models import build_feature_sets, load_models, save_models, train_all_models


class ProjectSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviews = [
            "The room was filthy and the staff were rude.",
            "Awful hotel. I will never return!",
            "The bed was uncomfortable and the service was slow.",
            "A terrible stay with a dirty bathroom.",
            "The location was noisy and breakfast was bad.",
            "Our reservation was lost; this was disappointing.",
            "The staff were friendly and the room was very clean.",
            "Wonderful stay! We loved the comfortable bed.",
            "Excellent hotel with a beautiful view.",
            "Great service, clean rooms, and a perfect location.",
            "Amazing breakfast and helpful people.",
            "We had a fantastic and relaxing vacation.",
        ]
        self.labels = np.asarray([0] * 6 + [1] * 6, dtype=int)

    def test_labeling_and_stratified_split(self) -> None:
        frame = pd.DataFrame(
            {
                "review": self.reviews[:10],
                "rating": [1, 2, 1, 2, 2, 3, 4, 5, 4, 5],
            }
        )
        labeled = create_sentiment_labels(frame)
        self.assertNotIn(3, labeled["rating"].tolist())
        train, validation, test = split_data(labeled, random_state=9)
        self.assertEqual(len(train) + len(validation) + len(test), len(labeled))
        self.assertEqual(set(train["sentiment"]), {0, 1})

    def test_full_model_api(self) -> None:
        self.assertEqual(extract_all_features(self.reviews[0]).shape, (15,))
        train_reviews, validation_reviews, test_reviews = (
            self.reviews[:8],
            self.reviews[8:10],
            self.reviews[10:],
        )
        y_train, y_validation, y_test = self.labels[:8], self.labels[8:10], self.labels[10:]

        doc2vec = train_doc2vec(train_reviews, vector_size=12, epochs=8, min_count=1)
        feature_sets = build_feature_sets(
            build_feature_matrices(train_reviews),
            build_feature_matrices(validation_reviews),
            build_feature_matrices(test_reviews),
            get_doc2vec_vectors(doc2vec, train_reviews, training=True),
            get_doc2vec_vectors(doc2vec, validation_reviews),
            get_doc2vec_vectors(doc2vec, test_reviews),
        )
        models = train_all_models(feature_sets, y_train, y_validation)
        self.assertEqual(len(models["models"]), 10)

        evaluation = evaluate_all_models(models, feature_sets, y_test)
        self.assertEqual(len(evaluation["metrics"]), 10)
        self.assertIn(
            predict_sentiment(
                test_reviews[0], "logistic_regression__basic_linguistic", models
            ),
            {"Positive", "Negative"},
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            save_doc2vec(doc2vec, output / "doc2vec.model")
            loaded_doc2vec = load_doc2vec(output / "doc2vec.model")
            self.assertEqual(loaded_doc2vec.vector_size, 12)
            self.assertEqual(
                get_doc2vec_vectors(loaded_doc2vec, train_reviews, training=True).shape,
                (len(train_reviews), 12),
            )
            save_models(models, output / "classifiers.joblib")
            loaded_models = load_models(output / "classifiers.joblib")
            self.assertEqual(len(loaded_models["models"]), 10)
            self.assertIn(
                predict_sentiment(
                    test_reviews[0], "logistic_regression__basic_linguistic", loaded_models
                ),
                {"Positive", "Negative"},
            )
            save_metrics(evaluation, output / "metrics.csv")
            save_confusion_matrices(evaluation, output / "matrices")
            save_error_analysis(test_reviews, y_test, evaluation, output / "errors.csv")
            save_feature_contributions(models, output / "coefficients.csv")
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertEqual(len(list((output / "matrices").glob("*.csv"))), 10)
            self.assertTrue((output / "errors.csv").is_file())
            self.assertTrue((output / "coefficients.csv").is_file())


if __name__ == "__main__":
    unittest.main()

