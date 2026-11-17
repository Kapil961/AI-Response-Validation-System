
import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyVectorStore:
    """
    Persistent vector store using NumPy.

    Stores normalized embedding vectors alongside their
    document and chunk metadata.

    Supports exact cosine similarity search.
    """

    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vectors = np.empty(
            (0, 384),
            dtype=np.float32,
        )

        self.records: list[dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        records: list[dict[str, Any]],
    ) -> None:
        """Add embedding vectors and their metadata."""

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != 384:
            raise ValueError(
                "Embedding dimension must be 384."
            )

        if len(embeddings) != len(records):
            raise ValueError(
                "Number of embeddings and records must match."
            )

        if len(embeddings) == 0:
            raise ValueError("Cannot add an empty batch.")

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain invalid values.")

        norms = np.linalg.norm(embeddings, axis=1)

        if np.any(norms == 0):
            raise ValueError("Zero vectors cannot be stored.")

        normalized_embeddings = (
            embeddings / norms[:, np.newaxis]
        )

        existing_ids = {
            record["chunk_id"] for record in self.records
        }

        incoming_ids = [
            record.get("chunk_id") for record in records
        ]

        if any(not chunk_id for chunk_id in incoming_ids):
            raise ValueError("Every record must have a chunk ID.")

        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("Duplicate chunk IDs in incoming batch.")

        if existing_ids.intersection(incoming_ids):
            raise ValueError("Chunk ID already exists in vector store.")

        for record in records:
            if not isinstance(record, dict):
                raise TypeError("Every record must be a dictionary.")

            if not isinstance(record.get("text"), str):
                raise ValueError("Every record must contain text.")

            if not isinstance(record.get("metadata", {}), dict):
                raise ValueError("Record metadata must be a dictionary.")

        self.vectors = np.vstack(
            [self.vectors, normalized_embeddings]
        )

        self.records.extend(records)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most similar stored chunks."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if len(self.records) == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be one-dimensional.")

        if query_embedding.shape[0] != 384:
            raise ValueError(
                "Query embedding dimension must be 384."
            )

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Query embedding contains invalid values.")

        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")

        query_embedding = query_embedding / query_norm

        similarities = self.vectors @ query_embedding

        result_count = min(top_k, len(self.records))

        ranked_indices = np.argsort(-similarities)[:result_count]

        results = []

        for index in ranked_indices:
            results.append({
                "chunk_id": self.records[index]["chunk_id"],
                "document_id": self.records[index].get("document_id"),
                "text": self.records[index]["text"],
                "metadata": self.records[index].get("metadata", {}),
                "similarity": float(similarities[index]),
            })

        return results

    def save(self) -> None:
        """Persist vectors and metadata to disk."""

        vectors_path = self.storage_path / "vectors.npy"
        records_path = self.storage_path / "records.json"

        np.save(vectors_path, self.vectors)

        with records_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.records,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self) -> None:
        """Load previously saved vectors and metadata."""

        vectors_path = self.storage_path / "vectors.npy"
        records_path = self.storage_path / "records.json"

        if not vectors_path.exists() or not records_path.exists():
            raise FileNotFoundError(
                "Vector store files were not found."
            )

        vectors = np.load(
            vectors_path,
            allow_pickle=False,
        )

        with records_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)

        if vectors.ndim != 2 or vectors.shape[1] != 384:
            raise ValueError("Stored vectors have an invalid shape.")

        if len(vectors) != len(records):
            raise ValueError(
                "Stored vector and record counts do not match."
            )

        if not np.all(np.isfinite(vectors)):
            raise ValueError("Stored vectors contain invalid values.")

        self.vectors = vectors.astype(np.float32, copy=False)
        self.records = records

    def count(self) -> int:
        """Return the number of stored vectors."""

        return len(self.records)
