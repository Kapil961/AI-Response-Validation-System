import ast
import csv
import json
import hashlib
from pathlib import Path
from typing import Any


class DatasetLoader:
    """
    Loads and standardizes TruthfulQA and SQuAD datasets.
    """

    @staticmethod
    def _parse_answers(value: Any) -> list[str]:
        """Convert an answer field into a list of answer strings."""

        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        if not isinstance(value, str) or not value.strip():
            return []

        value = value.strip()

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = value

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

        if isinstance(parsed, str):
            return [
                item.strip()
                for item in parsed.split(";")
                if item.strip()
            ]

        return [str(parsed).strip()] if parsed else []

    @staticmethod
    def normalize_record(
        dataset: str,
        question: str,
        answer: str,
        context: str = "",
        metadata: dict | None = None,
    ) -> dict:
        """Create a common record structure for all datasets."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        return {
            "dataset": dataset,
            "question": question.strip(),
            "answer": answer.strip() if answer else "",
            "context": context.strip() if context else "",
            "metadata": metadata or {},
        }

    @classmethod
    def load_truthfulqa_csv(cls, file_path: str | Path) -> list[dict]:
        """Load and normalize the TruthfulQA CSV dataset."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        records = []

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError("CSV file has no header.")

            for row_number, row in enumerate(reader, start=2):

                normalized_row = {
                    str(key).strip().lower(): value
                    for key, value in row.items()
                    if key is not None
                }

                question = normalized_row.get("question", "")

                if not question or not question.strip():
                    continue

                best_answer = normalized_row.get("best answer", "")

                correct_answers = cls._parse_answers(
                    normalized_row.get("correct answers", "")
                )

                answer = (
                    best_answer.strip()
                    if best_answer and best_answer.strip()
                    else correct_answers[0] if correct_answers else ""
                )

                metadata = {
                    "row_number": row_number,
                    "category": normalized_row.get("category", ""),
                    "type": normalized_row.get("type", ""),
                    "correct_answers": correct_answers,
                    "incorrect_answers": cls._parse_answers(
                        normalized_row.get("incorrect answers", "")
                    ),
                }

                records.append(
                    cls.normalize_record(
                        dataset="truthfulqa",
                        question=question,
                        answer=answer,
                        context="",
                        metadata=metadata,
                    )
                )

        return records

    @classmethod
    def load_squad_json(cls, file_path: str | Path) -> list[dict]:
        """Load and normalize the SQuAD JSON dataset."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            dataset_data = json.load(file)

        articles = dataset_data.get("data")

        if not isinstance(articles, list):
            raise ValueError(
                "Invalid SQuAD format: expected a 'data' list."
            )

        records = []

        for article in articles:

            title = article.get("title", "")

            for paragraph_index, paragraph in enumerate(
                article.get("paragraphs", [])
            ):

                context = paragraph.get("context", "")

                for qa in paragraph.get("qas", []):

                    question = qa.get("question", "")

                    if not question or not question.strip():
                        continue

                    answers = qa.get("answers", [])

                    answer = (
                        answers[0].get("text", "")
                        if answers else ""
                    )

                    metadata = {
                        "id": qa.get("id", ""),
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "is_impossible": qa.get(
                            "is_impossible",
                            False,
                        ),
                        "answers": answers,
                    }

                    records.append(
                        cls.normalize_record(
                            dataset="squad",
                            question=question,
                            answer=answer,
                            context=context,
                            metadata=metadata,
                        )
                    )

        return records

    @classmethod
    def load_squad_documents(
        cls,
        file_path: str | Path,
        split: str,
    ) -> list[dict]:
        """
        Extract unique SQuAD paragraph documents for RAG ingestion.

        Each paragraph is stored once, even if multiple questions
        refer to the same context.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            dataset_data = json.load(file)

        articles = dataset_data.get("data")

        if not isinstance(articles, list):
            raise ValueError(
                "Invalid SQuAD format: expected a 'data' list."
            )

        documents = []
        seen_contexts = set()

        for article in articles:
            title = article.get("title", "")

            for paragraph_index, paragraph in enumerate(
                article.get("paragraphs", [])
            ):
                context = paragraph.get("context", "")

                if not isinstance(context, str) or not context.strip():
                    continue

                context = context.strip()

                # Identify duplicate paragraphs using title and context.
                deduplication_key = (title, context)

                if deduplication_key in seen_contexts:
                    continue

                seen_contexts.add(deduplication_key)

                # Generate a deterministic document ID.
                unique_content = (
                    f"{split}\0{title}\0{context}"
                )

                document_id = hashlib.sha256(
                    unique_content.encode("utf-8")
                ).hexdigest()[:16]

                documents.append({
                    "document_id": document_id,
                    "text": context,
                    "metadata": {
                        "dataset": "squad",
                        "split": split,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "source_file": path.name,
                    },
                })

        return documents
