
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Generates normalized semantic embeddings for text.

    Uses the all-MiniLM-L6-v2 model, which produces
    384-dimensional sentence embeddings.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
    ):
        if not model_name.strip():
            raise ValueError("Model name cannot be empty.")

        self.model_name = model_name
        self.device = device

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        actual_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        if actual_dimension != self.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {self.EMBEDDING_DIMENSION} dimensions, "
                f"but model returned {actual_dimension}."
            )

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        """Validate and normalize input text."""

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise ValueError("At least one text is required.")

        validated_texts = []

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Every input must be a string.")

            cleaned_text = text.strip()

            if not cleaned_text:
                raise ValueError("Input text cannot be empty.")

            validated_texts.append(cleaned_text)

        return validated_texts

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Convert text into normalized embedding vectors.
        """

        validated_texts = self._validate_texts(texts)

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        embeddings = self.model.encode(
            validated_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.EMBEDDING_DIMENSION:
            raise ValueError("Unexpected embedding dimension.")

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Generate an embedding for one text."""

        return self.encode([text])[0]

    def get_embedding_dimension(self) -> int:
        """Return the embedding vector dimension."""

        return self.EMBEDDING_DIMENSION
