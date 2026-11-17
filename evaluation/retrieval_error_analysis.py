
from collections import Counter, defaultdict
from typing import Any


class RetrievalErrorAnalyzer:
    """
    Analyzes retrieval evaluation results and identifies
    question-level and article-level retrieval failure patterns.
    """

    def __init__(self, top_k: int = 5):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self.top_k = top_k

    @staticmethod
    def _is_relevant(
        result: dict[str, Any],
        expected_title: str,
        expected_paragraph_index: int,
    ) -> bool:
        metadata = result.get("metadata", {})

        return (
            metadata.get("title") == expected_title
            and metadata.get("paragraph_index") == expected_paragraph_index
        )

    def analyze_question(
        self,
        evaluation_result: dict[str, Any],
    ) -> dict[str, Any]:

        expected_title = evaluation_result["expected_title"]
        expected_paragraph_index = evaluation_result[
            "expected_paragraph_index"
        ]

        results = evaluation_result.get("results", [])

        relevant_ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if self._is_relevant(
                result,
                expected_title,
                expected_paragraph_index,
            )
        ]

        if not relevant_ranks:
            category = "retrieval_failure"
            first_relevant_rank = None

        else:
            first_relevant_rank = relevant_ranks[0]

            if first_relevant_rank == 1:
                category = "rank_1_success"
            else:
                category = "lower_rank_success"

        return {
            "question": evaluation_result["question"],
            "expected_title": expected_title,
            "expected_paragraph_index": expected_paragraph_index,
            "category": category,
            "first_relevant_rank": first_relevant_rank,
            "relevant_ranks": relevant_ranks,
            "retrieved_count": len(results),
        }

    def analyze_dataset(
        self,
        evaluation_results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not evaluation_results:
            raise ValueError("Evaluation results cannot be empty.")

        analyzed_questions = [
            self.analyze_question(result)
            for result in evaluation_results
        ]

        category_counts = Counter(
            result["category"]
            for result in analyzed_questions
        )

        total = len(analyzed_questions)

        # Group question-level results by expected source article.
        article_groups = defaultdict(list)

        for result in analyzed_questions:
            article_groups[result["expected_title"]].append(result)

        article_statistics = []

        for title, questions in article_groups.items():
            article_total = len(questions)

            counts = Counter(
                question["category"]
                for question in questions
            )

            failure_count = counts["retrieval_failure"]

            article_statistics.append({
                "title": title,
                "total_questions": article_total,
                "rank_1_success": counts["rank_1_success"],
                "lower_rank_success": counts["lower_rank_success"],
                "retrieval_failure": failure_count,
                "retrieval_failure_rate": (
                    failure_count / article_total
                ),
            })

        # Sort articles by failure count, then failure rate.
        article_statistics.sort(
            key=lambda article: (
                article["retrieval_failure"],
                article["retrieval_failure_rate"],
            ),
            reverse=True,
        )

        summary = {
            "total_questions": total,
            "top_k": self.top_k,
            "rank_1_success": category_counts["rank_1_success"],
            "lower_rank_success": category_counts["lower_rank_success"],
            "retrieval_failure": category_counts["retrieval_failure"],
        }

        summary["rank_1_success_rate"] = (
            summary["rank_1_success"] / total
        )

        summary["lower_rank_success_rate"] = (
            summary["lower_rank_success"] / total
        )

        summary["retrieval_failure_rate"] = (
            summary["retrieval_failure"] / total
        )

        summary["total_articles"] = len(article_statistics)

        summary["article_statistics"] = article_statistics

        summary["questions"] = analyzed_questions

        return summary
