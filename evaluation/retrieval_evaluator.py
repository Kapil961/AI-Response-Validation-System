
from typing import Any

from rag.retrieval.retrieval_engine import RetrievalEngine


class RetrievalEvaluator:
    """
    Evaluates semantic retrieval using SQuAD validation questions.
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        top_k: int = 5,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self.retrieval_engine = retrieval_engine
        self.top_k = top_k

    @staticmethod
    def _is_relevant(
        retrieved_result: dict[str, Any],
        expected_title: str,
        expected_paragraph_index: int,
    ) -> bool:
        metadata = retrieved_result.get("metadata", {})

        return (
            metadata.get("title") == expected_title
            and metadata.get("paragraph_index") == expected_paragraph_index
        )

    def evaluate_question(
        self,
        question_record: dict[str, Any],
    ) -> dict[str, Any]:
        question = question_record.get("question", "")
        metadata = question_record.get("metadata", {})

        expected_title = metadata.get("title")
        expected_paragraph_index = metadata.get("paragraph_index")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not expected_title:
            raise ValueError("Expected source title is missing.")

        if expected_paragraph_index is None:
            raise ValueError("Expected paragraph index is missing.")

        results = self.retrieval_engine.retrieve(
            question,
            top_k=self.top_k,
        )

        relevance = [
            self._is_relevant(
                result,
                expected_title,
                expected_paragraph_index,
            )
            for result in results
        ]

        relevant_count = sum(relevance)

        recall_at_k = float(relevant_count > 0)
        precision_at_k = relevant_count / self.top_k

        reciprocal_rank = 0.0

        for rank, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break

        return {
            "question": question,
            "expected_title": expected_title,
            "expected_paragraph_index": expected_paragraph_index,
            "retrieved_count": len(results),
            "relevant_count": relevant_count,
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "reciprocal_rank": reciprocal_rank,
            "results": results,
        }

    def evaluate_dataset(
        self,
        question_records: list[dict[str, Any]],
        limit: int | None = None,
    ) -> dict[str, Any]:

        if not question_records:
            raise ValueError("Question dataset cannot be empty.")

        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero.")

        records = (
            question_records[:limit]
            if limit is not None
            else question_records
        )

        recall_scores = []
        precision_scores = []
        reciprocal_ranks = []
        question_results = []

        for index, record in enumerate(records, start=1):
            result = self.evaluate_question(record)

            recall_scores.append(result["recall_at_k"])
            precision_scores.append(result["precision_at_k"])
            reciprocal_ranks.append(result["reciprocal_rank"])

            # Preserve individual results for retrieval error analysis.
            question_results.append(result)

            if index % 100 == 0 or index == len(records):
                print(f"Evaluated {index}/{len(records)} questions")

        total = len(records)

        return {
            "total_questions": total,
            "top_k": self.top_k,
            "recall_at_k": sum(recall_scores) / total,
            "precision_at_k": sum(precision_scores) / total,
            "mrr": sum(reciprocal_ranks) / total,
            "question_results": question_results,
        }
