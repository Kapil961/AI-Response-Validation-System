
import re
from collections import Counter
from typing import Optional


class AccuracyJudge:

    def tokenize(self, text: str) -> list:
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def evaluate(
        self,
        response: str,
        reference_answer: Optional[str] = None
    ) -> dict:

        if not reference_answer or not reference_answer.strip():
            return {
                "judge_name": "Accuracy Judge",
                "score": None,
                "passed": None,
                "explanation": (
                    "Accuracy cannot be estimated because no "
                    "reference answer was provided."
                ),
                "evidence": None,
                "status": "insufficient_evidence"
            }

        response_tokens = self.tokenize(response)
        reference_tokens = self.tokenize(reference_answer)

        if not response_tokens or not reference_tokens:
            return {
                "judge_name": "Accuracy Judge",
                "score": 0.0,
                "passed": False,
                "explanation": "Response or reference answer is empty.",
                "evidence": None,
                "status": "evaluated"
            }

        response_counts = Counter(response_tokens)
        reference_counts = Counter(reference_tokens)

        common = sum(
            (response_counts & reference_counts).values()
        )

        precision = common / len(response_tokens)
        recall = common / len(reference_tokens)

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (
                2 * precision * recall
            ) / (precision + recall)

        score = round(f1_score * 100, 2)

        return {
            "judge_name": "Accuracy Judge",
            "score": score,
            "passed": score >= 70,
            "explanation": (
                "The score is based on token-level precision, "
                "recall, and F1 overlap with the reference answer. "
                "It is a lexical baseline and does not independently "
                "verify factual correctness."
            ),
            "evidence": (
                f"Matched tokens: {common}; "
                f"Response tokens: {len(response_tokens)}; "
                f"Reference tokens: {len(reference_tokens)}"
            ),
            "status": "evaluated"
        }
