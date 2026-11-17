
import hashlib
import re
from typing import Any


class TextPreprocessor:
    """
    Cleans text and splits documents into overlapping chunks
    while preserving source metadata.
    """

    def __init__(
        self,
        chunk_size: int = 150,
        chunk_overlap: int = 30,
    ):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize whitespace without removing meaningful content
        or changing punctuation.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text

    def chunk_document(self, document: dict[str, Any]) -> list[dict]:
        """
        Split a document into overlapping word-based chunks.

        Each chunk retains its original document metadata and
        receives a deterministic unique chunk ID.
        """

        document_id = document.get("document_id")
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        if not document_id:
            raise ValueError("Document ID cannot be empty.")

        if not isinstance(text, str):
            raise TypeError("Document text must be a string.")

        if not isinstance(metadata, dict):
            raise TypeError("Document metadata must be a dictionary.")

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            return []

        words = cleaned_text.split()

        chunks = []

        step_size = self.chunk_size - self.chunk_overlap

        for start in range(0, len(words), step_size):
            end = min(start + self.chunk_size, len(words))

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk_index = len(chunks)

            chunk_hash_input = (
                f"{document_id}:{chunk_index}:{chunk_text}"
            )

            chunk_id = hashlib.sha256(
                chunk_hash_input.encode("utf-8")
            ).hexdigest()[:16]

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_words),
                    "start_word": start,
                    "end_word": end,
                },
            })

            if end == len(words):
                break

        return chunks
