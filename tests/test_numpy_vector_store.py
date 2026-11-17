
import numpy as np
import pytest

from rag.vector_store.numpy_vector_store import NumpyVectorStore


@pytest.fixture
def vector_store(tmp_path):
    return NumpyVectorStore(tmp_path / "vector_db")


def create_records(count):
    return [
        {
            "chunk_id": f"chunk-{index}",
            "document_id": f"doc-{index}",
            "text": f"Sample document text {index}",
            "metadata": {
                "dataset": "squad",
                "split": "train",
            },
        }
        for index in range(count)
    ]


def test_add_vectors(vector_store):
    embeddings = np.random.rand(3, 384).astype(np.float32)

    vector_store.add(embeddings, create_records(3))

    assert vector_store.count() == 3
    assert vector_store.vectors.shape == (3, 384)


def test_search_returns_top_k_results(vector_store):
    embeddings = np.zeros((3, 384), dtype=np.float32)

    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    embeddings[2, 2] = 1.0

    vector_store.add(embeddings, create_records(3))

    query = np.zeros(384, dtype=np.float32)
    query[1] = 1.0

    results = vector_store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_search_limits_results_to_available_records(vector_store):
    embeddings = np.random.rand(2, 384).astype(np.float32)

    vector_store.add(embeddings, create_records(2))

    results = vector_store.search(
        np.random.rand(384).astype(np.float32),
        top_k=10,
    )

    assert len(results) == 2


def test_save_and_load(vector_store, tmp_path):
    embeddings = np.random.rand(3, 384).astype(np.float32)

    vector_store.add(embeddings, create_records(3))
    vector_store.save()

    new_store = NumpyVectorStore(tmp_path / "vector_db")
    new_store.load()

    assert new_store.count() == 3
    assert np.allclose(new_store.vectors, vector_store.vectors)
    assert new_store.records == vector_store.records


def test_add_rejects_wrong_embedding_dimension(vector_store):
    embeddings = np.random.rand(2, 128).astype(np.float32)

    with pytest.raises(ValueError):
        vector_store.add(embeddings, create_records(2))


def test_add_rejects_mismatched_record_count(vector_store):
    embeddings = np.random.rand(2, 384).astype(np.float32)

    with pytest.raises(ValueError):
        vector_store.add(embeddings, create_records(1))


def test_add_rejects_duplicate_chunk_ids(vector_store):
    embeddings = np.random.rand(2, 384).astype(np.float32)

    records = create_records(2)
    records[1]["chunk_id"] = records[0]["chunk_id"]

    with pytest.raises(ValueError):
        vector_store.add(embeddings, records)


def test_search_rejects_invalid_top_k(vector_store):
    with pytest.raises(ValueError):
        vector_store.search(
            np.random.rand(384).astype(np.float32),
            top_k=0,
        )


def test_search_rejects_wrong_query_dimension(vector_store):
    embeddings = np.random.rand(1, 384).astype(np.float32)

    vector_store.add(embeddings, create_records(1))

    with pytest.raises(ValueError):
        vector_store.search(
            np.random.rand(128).astype(np.float32)
        )


def test_empty_store_returns_no_results(vector_store):
    results = vector_store.search(
        np.random.rand(384).astype(np.float32)
    )

    assert results == []
