
import re

from collections import Counter
from typing import Optional


class AccuracyJudge:

    def __init__(self, passing_threshold: float = 70.0):
        self.passing_threshold = passing_threshold

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
                "classification": "insufficient_evidence",
                "explanation": (
                    "Accuracy cannot be estimated because no "
                    "reference answer was provided."
                ),
                "evidence": None,
                "matched_concepts": [],
                "missing_concepts": [],
                "status": "insufficient_evidence"
            }

        response_tokens = self.tokenize(response)
        reference_tokens = self.tokenize(reference_answer)

        if not response_tokens or not reference_tokens:
            return {
                "judge_name": "Accuracy Judge",
                "score": 0.0,
                "passed": False,
                "classification": "incorrect",
                "explanation": (
                    "Response or reference answer contains "
                    "no meaningful tokens."
                ),
                "evidence": None,
                "matched_concepts": [],
                "missing_concepts": [],
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

        if score >= 85:
            classification = "correct"
        elif score >= 50:
            classification = "partially_correct"
        else:
            classification = "incorrect"

        matched = sorted(
            set(response_tokens).intersection(reference_tokens)
        )

        missing = sorted(
            set(reference_tokens) - set(response_tokens)
        )

        explanation = (
            f"The response achieved {score}% lexical F1 overlap "
            f"with the reference answer. "
            f"Classification: {classification}. "
            "This comparison does not independently verify "
            "factual correctness or detect all contradictions."
        )

        return {
            "judge_name": "Accuracy Judge",
            "score": score,
            "passed": score >= self.passing_threshold,
            "classification": classification,
            "explanation": explanation,
            "evidence": (
                f"Matched tokens: {common}; "
                f"Response tokens: {len(response_tokens)}; "
                f"Reference tokens: {len(reference_tokens)}"
            ),
            "matched_concepts": matched,
            "missing_concepts": missing,
            "status": "evaluated"
        }
