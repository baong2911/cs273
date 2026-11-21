"""Train, persist, and use the project's Gensim Doc2Vec representation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

import numpy as np

try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "Gensim is required for Doc2Vec. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc

try:
    from .nlp_utils import word_tokenize
except ImportError:  # pragma: no cover - only used for direct script execution
    from nlp_utils import word_tokenize


DEFAULT_VECTOR_SIZE = 100
DEFAULT_EPOCHS = 40
DEFAULT_RANDOM_STATE = 42


def _as_review_list(reviews: Iterable[object]) -> List[str]:
    if isinstance(reviews, str):
        return [reviews]
    return ["" if review is None else str(review) for review in reviews]


def _tokens(review: object) -> List[str]:
    # Lowercasing is consistent between training, evaluation, and GUI input.
    return [token.casefold() for token in word_tokenize(review) if token.strip()]


def train_doc2vec(
    training_reviews: Iterable[object],
    vector_size: int = DEFAULT_VECTOR_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    random_state: int = DEFAULT_RANDOM_STATE,
    window: int = 5,
    min_count: int = 2,
    workers: int = 1,
) -> Doc2Vec:
    """Train a Doc2Vec model on training reviews only.

    ``workers=1`` is the default because it makes the training step much more
    reproducible for a course experiment.  Validation and test reviews are
    never included in the tagged training corpus.
    """

    review_list = _as_review_list(training_reviews)
    if not review_list:
        raise ValueError("Cannot train Doc2Vec without at least one training review.")
    if vector_size <= 0 or epochs <= 0 or window <= 0 or min_count <= 0 or workers <= 0:
        raise ValueError("Doc2Vec vector_size, epochs, window, min_count, and workers must be positive.")

    tagged_documents = [
        TaggedDocument(words=_tokens(review), tags=[f"TRAIN_{index}"])
        for index, review in enumerate(review_list)
    ]
    if not any(document.words for document in tagged_documents):
        raise ValueError("Training reviews did not contain any tokenizable text.")

    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=random_state,
        dm=1,
    )
    model.build_vocab(tagged_documents)
    if model.corpus_count == 0 or len(model.wv) == 0:
        raise ValueError("Doc2Vec vocabulary is empty; lower min_count or inspect the input reviews.")
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

    # Persist enough metadata to safely retrieve document vectors for exactly
    # the original training sequence after loading the model from disk.
    model.cs273_training_count = len(review_list)
    model.cs273_training_tag_prefix = "TRAIN_"
    model.cs273_inference_epochs = max(20, epochs)
    model.cs273_inference_seed = random_state
    return model


def infer_review_vector(model: Doc2Vec, review: object) -> np.ndarray:
    """Infer one fixed-length Doc2Vec vector for a new review."""

    tokens = _tokens(review)
    if not tokens:
        return np.zeros(model.vector_size, dtype=float)

    # A fixed seed prevents avoidable run-to-run differences in validation,
    # test, and GUI vectors while leaving trained document vectors unchanged.
    inference_seed = int(getattr(model, "cs273_inference_seed", DEFAULT_RANDOM_STATE))
    model.random.seed(inference_seed)
    epochs = int(getattr(model, "cs273_inference_epochs", DEFAULT_EPOCHS))
    return np.asarray(model.infer_vector(tokens, epochs=epochs), dtype=float)


def get_doc2vec_vectors(
    model: Doc2Vec, reviews: Iterable[object], training: bool = False
) -> np.ndarray:
    """Return vectors for a review collection.

    Set ``training=True`` only when *reviews* is the same ordered training
    sequence passed to ``train_doc2vec``.  That retrieves learned document
    vectors; validation, test, and GUI text must use inferred vectors.
    """

    review_list = _as_review_list(reviews)
    if not review_list:
        return np.empty((0, model.vector_size), dtype=float)

    if training:
        expected_count = getattr(model, "cs273_training_count", None)
        prefix = str(getattr(model, "cs273_training_tag_prefix", "TRAIN_"))
        if expected_count != len(review_list):
            raise ValueError(
                "Training-vector retrieval requires the original complete ordered "
                f"training set ({expected_count} reviews), not {len(review_list)} reviews."
            )
        try:
            return np.vstack(
                [np.asarray(model.dv[f"{prefix}{index}"], dtype=float) for index in range(len(review_list))]
            )
        except KeyError as exc:
            raise ValueError("The Doc2Vec model is missing expected training document tags.") from exc

    return np.vstack([infer_review_vector(model, review) for review in review_list])


def save_doc2vec(model: Doc2Vec, path: Union[str, Path]) -> None:
    """Save a trained Doc2Vec model, creating its parent directory if needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))


def load_doc2vec(path: Union[str, Path]) -> Doc2Vec:
    """Load a previously trained Doc2Vec model with an informative error."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Doc2Vec model not found at {source}. Run `python main.py` before launching the GUI."
        )
    return Doc2Vec.load(str(source))

