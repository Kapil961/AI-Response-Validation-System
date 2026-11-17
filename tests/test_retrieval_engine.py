
import pytest

from rag.retrieval import retrieval_engine


class MockEmbeddingModel:
    def __init__(self, device="cpu"):
        self.device = device

    def encode_single(self, text):
        return [0.1, 0.2, 0.3]


class MockVectorStore:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.loaded = False

    def load(self):
        self.loaded = True

    def search(self, query_embedding, top_k=5):
        results = [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "text": "Artificial intelligence is a field of computer science.",
                "metadata": {"title": "AI"},
                "similarity": 0.95,
            },
            {
                "chunk_id": "chunk_2",
                "document_id": "doc_2",
                "text": "Machine learning is a subset of AI.",
                "metadata": {"title": "ML"},
                "similarity": 0.80,
            },
            {
                "chunk_id": "chunk_3",
                "document_id": "doc_3",
                "text": "The history of mathematics.",
                "metadata": {"title": "Math"},
                "similarity": 0.35,
            },
        ]

        return results[:top_k]


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setattr(
        retrieval_engine,
        "EmbeddingModel",
        MockEmbeddingModel,
    )

    monkeypatch.setattr(
        retrieval_engine,
        "NumpyVectorStore",
        MockVectorStore,
    )

    return retrieval_engine.RetrievalEngine(
        storage_path=tmp_path,
        top_k=3,
        similarity_threshold=0.0,
    )


def test_retrieval_returns_top_k_results(engine):
    results = engine.retrieve("What is artificial intelligence?")

    assert len(results) == 3
    assert results[0]["similarity"] == 0.95


def test_retrieval_respects_top_k(engine):
    results = engine.retrieve(
        "What is artificial intelligence?",
        top_k=2,
    )

    assert len(results) == 2


def test_retrieval_applies_similarity_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(
        retrieval_engine,
        "EmbeddingModel",
        MockEmbeddingModel,
    )

    monkeypatch.setattr(
        retrieval_engine,
        "NumpyVectorStore",
        MockVectorStore,
    )

    engine = retrieval_engine.RetrievalEngine(
        storage_path=tmp_path,
        top_k=3,
        similarity_threshold=0.70,
    )

    results = engine.retrieve("Explain artificial intelligence.")

    assert len(results) == 2
    assert all(result["similarity"] >= 0.70 for result in results)


def test_retrieval_rejects_empty_query(engine):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        engine.retrieve("   ")


def test_retrieval_rejects_non_string_query(engine):
    with pytest.raises(TypeError, match="Query must be a string"):
        engine.retrieve(123)


def test_retrieval_rejects_invalid_top_k(engine):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        engine.retrieve("Explain AI", top_k=0)


@pytest.mark.parametrize("top_k", [0, -1])
def test_constructor_rejects_invalid_top_k(tmp_path, top_k):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retrieval_engine.RetrievalEngine(
            storage_path=tmp_path,
            top_k=top_k,
        )


@pytest.mark.parametrize("threshold", [-1.1, 1.1])
def test_constructor_rejects_invalid_similarity_threshold(
    tmp_path,
    threshold,
):
    with pytest.raises(
        ValueError,
        match="Similarity threshold must be between -1 and 1",
    ):
        retrieval_engine.RetrievalEngine(
            storage_path=tmp_path,
            similarity_threshold=threshold,
        )
