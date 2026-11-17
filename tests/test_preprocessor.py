
import pytest

from rag.ingestion.preprocessor import TextPreprocessor


def test_clean_text_normalizes_whitespace():
    text = "  Artificial   Intelligence\n is\tpowerful.  "

    cleaned = TextPreprocessor.clean_text(text)

    assert cleaned == "Artificial Intelligence is powerful."


def test_chunk_document_preserves_short_document():
    processor = TextPreprocessor(
        chunk_size=150,
        chunk_overlap=30,
    )

    document = {
        "document_id": "doc-001",
        "text": "Artificial intelligence enables machines to learn.",
        "metadata": {
            "dataset": "squad",
            "split": "train",
            "title": "Artificial Intelligence",
        },
    }

    chunks = processor.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "doc-001"
    assert chunks[0]["text"] == document["text"]
    assert chunks[0]["metadata"]["dataset"] == "squad"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_chunk_document_creates_overlapping_chunks():
    processor = TextPreprocessor(
        chunk_size=10,
        chunk_overlap=2,
    )

    words = [f"word{i}" for i in range(18)]

    document = {
        "document_id": "doc-002",
        "text": " ".join(words),
        "metadata": {"split": "train"},
    }

    chunks = processor.chunk_document(document)

    assert len(chunks) == 2

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()

    assert len(first_words) == 10
    assert len(second_words) == 10

    assert first_words[-2:] == second_words[:2]


def test_chunk_document_returns_empty_for_empty_text():
    processor = TextPreprocessor()

    document = {
        "document_id": "doc-003",
        "text": "   ",
        "metadata": {},
    }

    assert processor.chunk_document(document) == []


def test_chunk_document_generates_unique_chunk_ids():
    processor = TextPreprocessor(
        chunk_size=5,
        chunk_overlap=1,
    )

    document = {
        "document_id": "doc-004",
        "text": "one two three four five six seven eight nine ten",
        "metadata": {"split": "train"},
    }

    chunks = processor.chunk_document(document)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_invalid_chunk_configuration():
    with pytest.raises(ValueError):
        TextPreprocessor(chunk_size=0)

    with pytest.raises(ValueError):
        TextPreprocessor(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError):
        TextPreprocessor(chunk_size=10, chunk_overlap=-1)
