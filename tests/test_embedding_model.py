
import numpy as np
import pytest

from rag.embeddings.embedding_model import EmbeddingModel


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel(device="cpu")


def test_model_dimension(embedding_model):
    assert embedding_model.get_embedding_dimension() == 384


def test_encode_single_text(embedding_model):
    embedding = embedding_model.encode_single(
        "Artificial intelligence enables machines to learn."
    )

    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32


def test_encode_multiple_texts(embedding_model):
    texts = [
        "Artificial intelligence is transforming technology.",
        "Machine learning is a branch of artificial intelligence.",
        "The weather is sunny today.",
    ]

    embeddings = embedding_model.encode(texts)

    assert embeddings.shape == (3, 384)
    assert embeddings.dtype == np.float32


def test_embeddings_are_normalized(embedding_model):
    texts = [
        "Artificial intelligence.",
        "Machine learning.",
    ]

    embeddings = embedding_model.encode(texts)

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_sentences_have_higher_similarity(embedding_model):
    texts = [
        "Artificial intelligence enables computers to learn.",
        "AI allows machines to learn and perform intelligent tasks.",
        "I enjoy eating fresh mangoes in summer.",
    ]

    embeddings = embedding_model.encode(texts)

    similarity = embeddings @ embeddings.T

    assert similarity[0, 1] > similarity[0, 2]


def test_encode_rejects_empty_input(embedding_model):
    with pytest.raises(ValueError):
        embedding_model.encode([])


def test_encode_rejects_blank_text(embedding_model):
    with pytest.raises(ValueError):
        embedding_model.encode(["   "])


def test_encode_rejects_invalid_batch_size(embedding_model):
    with pytest.raises(ValueError):
        embedding_model.encode(
            ["Test sentence"],
            batch_size=0,
        )
