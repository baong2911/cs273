"""Doc2Vec training, persistence, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import (
    DOC2VEC_EPOCHS,
    DOC2VEC_MIN_COUNT,
    DOC2VEC_VECTOR_SIZE,
    DOC2VEC_WINDOW,
    DOC2VEC_WORKERS,
    RANDOM_SEED,
)
from .text_processing import model_tokens


try:  # Keep imports lazy-friendly for modules/tests that do not need Doc2Vec.
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
except Exception:  # pragma: no cover - exercised only by broken optional installs
    Doc2Vec = None  # type: ignore[assignment]
    TaggedDocument = None  # type: ignore[assignment]


DEFAULT_DOC2VEC_CONFIG = {
    "vector_size": DOC2VEC_VECTOR_SIZE,
    "window": DOC2VEC_WINDOW,
    "min_count": DOC2VEC_MIN_COUNT,
    "workers": DOC2VEC_WORKERS,
    "epochs": DOC2VEC_EPOCHS,
}


def _require_gensim() -> None:
    if Doc2Vec is None or TaggedDocument is None:
        raise ImportError(
            "Gensim Doc2Vec is unavailable. Install requirements.txt and verify the "
            "NumPy/SciPy versions in the environment."
        )


def train_doc2vec(
    train_texts: Sequence[str],
    config: Mapping[str, int] | None = None,
    seed: int = RANDOM_SEED,
):
    """Train one Doc2Vec model on exactly the supplied texts."""

    _require_gensim()
    params = dict(DEFAULT_DOC2VEC_CONFIG)
    if config:
        params.update(config)
    documents = [
        TaggedDocument(words=model_tokens(text), tags=[f"DOC_{index}"])
        for index, text in enumerate(train_texts)
    ]
    if not documents:
        raise ValueError("Doc2Vec requires at least one training document")
    model = Doc2Vec(
        vector_size=int(params["vector_size"]),
        window=int(params["window"]),
        min_count=int(params["min_count"]),
        workers=int(params["workers"]),
        epochs=int(params["epochs"]),
        seed=int(seed),
        dm=1,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    model.project_config = {
        "vector_size": int(params["vector_size"]),
        "window": int(params["window"]),
        "min_count": int(params["min_count"]),
        "workers": int(params["workers"]),
        "epochs": int(params["epochs"]),
        "seed": int(seed),
    }
    return model


def training_vectors(model, document_count: int) -> np.ndarray:
    """Return learned document vectors in their original training order."""

    if document_count < 0:
        raise ValueError("document_count must be non-negative")
    if document_count == 0:
        return np.empty((0, int(model.vector_size)), dtype=np.float32)
    tags = [f"DOC_{index}" for index in range(document_count)]
    try:
        vectors = np.asarray(model.dv[tags], dtype=np.float32)
    except (KeyError, TypeError, IndexError):
        vectors = np.asarray(model.dv.vectors[:document_count], dtype=np.float32)
    return vectors


def infer_vectors(
    model,
    texts: Sequence[str],
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Infer vectors without updating the trained Doc2Vec model."""

    vectors = []
    epochs = int(getattr(model, "epochs", DOC2VEC_EPOCHS))
    for index, text in enumerate(texts):
        # Gensim's infer_vector uses the model RNG. Seeding it before every
        # document makes repeated inference reproducible without training on
        # validation, test, or user-entered text.
        random_state = getattr(model, "random", None)
        if random_state is not None and hasattr(random_state, "seed"):
            random_state.seed(int(seed) + index)
        try:
            inferred = model.infer_vector(model_tokens(text), epochs=epochs)
        except TypeError:
            # Small test doubles and older Gensim versions may not expose the
            # optional ``epochs`` keyword.
            inferred = model.infer_vector(model_tokens(text))
        vectors.append(np.asarray(inferred, dtype=np.float32))
    width = int(model.vector_size)
    return np.asarray(vectors, dtype=np.float32).reshape(len(vectors), width)


def save_doc2vec(model, path: Path | str) -> None:
    """Save a trained Doc2Vec model."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_doc2vec(path: Path | str):
    """Load a saved Doc2Vec model."""

    _require_gensim()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Doc2Vec model not found at {path}")
    return Doc2Vec.load(str(path))
