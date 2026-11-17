
from pathlib import Path
from typing import Any

from rag.embeddings.embedding_model import EmbeddingModel
from rag.vector_store.numpy_vector_store import NumpyVectorStore


class RetrievalEngine:
    """
    Semantic retrieval engine.

    Converts a user query into an embedding and retrieves
    the most relevant document chunks from the vector store.
    """

    def __init__(
        self,
        storage_path: str | Path,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "Similarity threshold must be between -1 and 1."
            )

        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self.embedding_model = EmbeddingModel(device="cpu")

        self.vector_store = NumpyVectorStore(storage_path)
        self.vector_store.load()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.
        """

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        result_limit = self.top_k if top_k is None else top_k

        if result_limit <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_model.encode_single(query)

        results = self.vector_store.search(
            query_embedding,
            top_k=result_limit,
        )

        filtered_results = [
            result
            for result in results
            if result["similarity"] >= self.similarity_threshold
        ]

        return filtered_results
