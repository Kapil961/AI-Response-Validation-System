
import numpy as np
import pytest

from rag.ingestion.indexing_pipeline import IndexingPipeline


@pytest.fixture
def pipeline(tmp_path):
    return IndexingPipeline(
        storage_path=tmp_path / "vector_db",
        chunk_size=10,
        chunk_overlap=2,
        batch_size=2,
    )


def sample_documents():
    return [
        {
            "document_id": "doc-001",
            "text": (
                "Artificial intelligence enables machines to "
                "learn from data and perform intelligent tasks."
            ),
            "metadata": {
                "dataset": "squad",
                "split": "train",
                "title": "Artificial Intelligence",
            },
        },
        {
            "document_id": "doc-002",
            "text": (
                "Machine learning is a branch of artificial "
                "intelligence that uses algorithms."
            ),
            "metadata": {
                "dataset": "squad",
                "split": "train",
                "title": "Machine Learning",
            },
        },
    ]


def test_index_documents(pipeline):
    result = pipeline.index_documents(sample_documents())

    assert result["documents_processed"] == 2
    assert result["chunks_generated"] > 0
    assert result["chunks_indexed"] == result["chunks_generated"]
    assert pipeline.vector_store.count() == result["chunks_indexed"]


def test_indexing_creates_persistent_files(pipeline):
    pipeline.index_documents(sample_documents())

    assert (pipeline.vector_store.storage_path / "vectors.npy").exists()
    assert (pipeline.vector_store.storage_path / "records.json").exists()


def test_indexing_preserves_document_metadata(pipeline):
    pipeline.index_documents(sample_documents())

    records = pipeline.vector_store.records

    assert all(record["metadata"]["dataset"] == "squad" for record in records)
    assert all(record["metadata"]["split"] == "train" for record in records)


def test_indexing_uses_valid_embedding_dimensions(pipeline):
    pipeline.index_documents(sample_documents())

    assert pipeline.vector_store.vectors.shape[1] == 384


def test_indexing_rejects_empty_documents(pipeline):
    with pytest.raises(ValueError):
        pipeline.index_documents([])


def test_indexing_rejects_documents_without_valid_text(pipeline):
    documents = [
        {
            "document_id": "doc-empty",
            "text": "   ",
            "metadata": {},
        }
    ]

    with pytest.raises(ValueError):
        pipeline.index_documents(documents)
