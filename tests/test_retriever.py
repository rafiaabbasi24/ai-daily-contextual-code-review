import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import the project's retrieval utilities.
# Adjust the import path according to the actual project layout.
# The tests assume the following public API:
# - `EmbeddingModel` with a constructor that optionally receives a model name,
#   and an `embed(texts: list[str]) -> np.ndarray` method.
# - `FAISSIndex` with a constructor `FAISSIndex(dimension: int)`,
#   an `add(vectors: np.ndarray)` method,
#   and a `search(query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]`
#   returning (ids, distances).
try:
    from retriever import EmbeddingModel, FAISSIndex  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Unable to import `EmbeddingModel` and `FAISSIndex` from the project. "
        "Make sure the test runner's PYTHONPATH includes the project root."
    ) from exc


@pytest.fixture(scope="module")
def embedding_model():
    """
    Fixture that creates a lightweight sentence‑transformers embedding model.
    The default model `all-MiniLM-L6-v2` is ~22 MB and fast enough for CI.
    """
    # The model name can be overridden via an environment variable for faster CI.
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return EmbeddingModel(model_name=model_name)


@pytest.fixture(scope="module")
def sample_corpus():
    """
    Small corpus of code‑related sentences used for indexing and retrieval.
    """
    return [
        "def add(a, b): return a + b",
        "def subtract(a, b): return a - b",
        "def multiply(a, b): return a * b",
        "def divide(a, b): return a / b if b != 0 else None",
        "class Calculator: pass",
    ]


@pytest.fixture(scope="module")
def corpus_embeddings(embedding_model, sample_corpus):
    """
    Pre‑compute embeddings for the sample corpus.
    """
    vectors = embedding_model.embed(sample_corpus)
    assert isinstance(vectors, np.ndarray), "Embedding output must be a NumPy array"
    assert vectors.ndim == 2, "Embedding output must be 2‑dimensional"
    return vectors


def test_embedding_shape(embedding_model):
    """
    Verify that the embedding model returns vectors of the expected dimensionality.
    """
    texts = ["print('Hello, world!')", "x = 1 + 2"]
    vectors = embedding_model.embed(texts)

    # The embedding dimension is model‑specific; we only assert shape consistency.
    assert isinstance(vectors, np.ndarray), "Embedding output must be a NumPy array"
    assert vectors.shape == (len(texts), vectors.shape[1]), "First dimension must match number of inputs"
    assert vectors.dtype == np.float32, "Embeddings should be float32 for FAISS compatibility"


def test_faiss_index_construction_and_search(corpus_embeddings):
    """
    Build a FAISS index from the pre‑computed corpus embeddings, add them,
    and ensure that a nearest‑neighbor query returns the expected document.
    """
    dim = corpus_embeddings.shape[1]
    index = FAISSIndex(dimension=dim)

    # Add vectors to the index; the method should accept a NumPy array.
    index.add(corpus_embeddings)

    # Query with the exact same vector as the first document.
    query_vec = corpus_embeddings[0:1]  # shape (1, dim)
    ids, distances = index.search(query_vec, k=1)

    # The nearest neighbor of a vector should be itself (distance ≈ 0).
    assert ids.shape == (1, 1), "Returned IDs shape mismatch"
    assert distances.shape == (1, 1), "Returned distances shape mismatch"
    assert ids[0, 0] == 0, "The closest document should be the first one (ID 0)"
    assert distances[0, 0] < 1e-5, "Distance to identical vector should be near zero"


def test_end_to_end_retrieval(embedding_model, sample_corpus, corpus_embeddings):
    """
    Perform a realistic retrieval: embed a query similar to one of the corpus
    entries and verify that the top‑ranked result matches the expected snippet.
    """
    dim = corpus_embeddings.shape[1]
    index = FAISSIndex(dimension=dim)
    index.add(corpus_embeddings)

    # Query that is a paraphrase of the second corpus entry.
    query = "def subtraction(x, y): return x - y"
    query_vec = embedding_model.embed([query])

    ids, distances = index.search(query_vec, k=3)

    # The expected nearest neighbor is the second entry (index 1).
    top_id = ids[0, 0]
    assert top_id == 1, f"Expected document ID 1, got {top_id}"

    # Ensure the distance is reasonable (i.e., not extremely large).
    top_distance = distances[0, 0]
    assert top_distance < 0.5, f"Unexpectedly large distance for a similar query: {top_distance}"


def test_faiss_index_persistence(tmp_path: Path, corpus_embeddings):
    """
    Verify that a FAISS index can be saved to disk and re‑loaded without loss
    of retrieval quality.
    """
    dim = corpus_embeddings.shape[1]
    index = FAISSIndex(dimension=dim)
    index.add(corpus_embeddings)

    # Save the index.
    index_path = tmp_path / "test.index"
    index.save(str(index_path))

    # Load a fresh index from the saved file.
    loaded_index = FAISSIndex.load(str(index_path))

    # Perform the same query as in `test_faiss_index_construction_and_search`.
    query_vec = corpus_embeddings[2:3]  # third document
    ids, distances = loaded_index.search(query_vec, k=1)

    assert ids[0, 0] == 2, "Loaded index should return the same nearest neighbor"
    assert distances[0, 0] < 1e-5, "Distance after reload should remain near zero"


def test_embedding_batch_consistency(embedding_model):
    """
    Ensure that embedding a batch of texts yields the same result as embedding them
    individually and stacking the results.
    """
    texts = [
        "def foo(): pass",
        "def bar(): return 42",
        "class Baz: ...",
    ]

    batch_vectors = embedding_model.embed(texts)
    individual_vectors = np.stack([embedding_model.embed([t])[0] for t in texts])

    # The two approaches should be numerically identical (within tolerance).
    np.testing.assert_allclose(batch_vectors, individual_vectors, rtol=1e-5, atol=1e-7)


def test_invalid_input_handling(embedding_model, tmp_path):
    """
    The embedding model and FAISS index should raise clear errors for malformed inputs.
    """
    # Empty string list should raise a ValueError.
    with pytest.raises(ValueError):
        embedding_model.embed([])

    # Non‑string elements in the list should raise a TypeError.
    with pytest.raises(TypeError):
        embedding_model.embed([123, "valid string"])

    # FAISS index constructed with a non‑positive dimension should raise.
    with pytest.raises(ValueError):
        FAISSIndex(dimension=0)

    # Adding vectors with mismatched dimensionality should raise.
    index = FAISSIndex(dimension=embedding_model.embed(["test"]).shape[1])
    wrong_dim_vec = np.random.rand(5, index.dimension + 1).astype(np.float32)
    with pytest.raises(ValueError):
        index.add(wrong_dim_vec)

    # Saving to a non‑existent directory should raise an OSError.
    index.add(embedding_model.embed(["test"]))
    invalid_path = tmp_path / "nonexistent_dir" / "index.faiss"
    with pytest.raises(OSError):
        index.save(str(invalid_path))