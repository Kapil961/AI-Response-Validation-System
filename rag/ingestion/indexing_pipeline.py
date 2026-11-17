
from pathlib import Path
from typing import Any

from rag.embeddings.embedding_model import EmbeddingModel
from rag.ingestion.dataset_loader import DatasetLoader
from rag.ingestion.preprocessor import TextPreprocessor
from rag.vector_store.numpy_vector_store import NumpyVectorStore


class IndexingPipeline:
    """
    End-to-end RAG indexing pipeline.

    Loads source documents, creates overlapping chunks,
    generates embeddings, and persists them in a vector store.
    """

    def __init__(
        self,
        storage_path: str | Path,
        chunk_size: int = 150,
        chunk_overlap: int = 30,
        batch_size: int = 32,
    ):
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")

        self.batch_size = batch_size

        self.preprocessor = TextPreprocessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embedding_model = EmbeddingModel(device="cpu")

        self.vector_store = NumpyVectorStore(storage_path)

    def index_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, int]:
        """
        Process documents in batches and store their embeddings.
        """

        if not documents:
            raise ValueError("No documents provided for indexing.")

        all_chunks = []

        for document in documents:
            chunks = self.preprocessor.chunk_document(document)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No valid chunks generated.")

        indexed_count = 0

        for start in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[
                start:start + self.batch_size
            ]

            texts = [
                chunk["text"] for chunk in batch
            ]

            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.batch_size,
            )

            self.vector_store.add(
                embeddings,
                batch,
            )

            indexed_count += len(batch)

            print(
                f"Indexed {indexed_count}/{len(all_chunks)} chunks"
            )

        self.vector_store.save()

        return {
            "documents_processed": len(documents),
            "chunks_generated": len(all_chunks),
            "chunks_indexed": indexed_count,
        }

    def index_squad_dataset(
        self,
        file_path: str | Path,
        split: str = "train",
    ) -> dict[str, int]:
        """
        Load SQuAD paragraph documents and index them.
        """

        documents = DatasetLoader.load_squad_documents(
            file_path,
            split=split,
        )

        return self.index_documents(documents)
