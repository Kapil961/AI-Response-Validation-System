
import re


class RelevanceJudge:

    def __init__(self, passing_threshold: float = 50.0):
        self.passing_threshold = passing_threshold

        self.stop_words = {
            "what", "is", "are", "the", "a", "an",
            "of", "in", "on", "to", "for", "and",
            "how", "why", "when", "where", "does",
            "do", "can", "could", "would", "should",
            "explain", "describe", "tell", "me", "about",
            "please", "give", "provide"
        }

    def tokenize(self, text: str) -> set:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return {
            word for word in words
            if word not in self.stop_words
        }

    def evaluate(self, question: str, response: str) -> dict:

        question_words = self.tokenize(question)
        response_words = self.tokenize(response)

        if not question_words:
            return {
                "judge_name": "Relevance Judge",
                "score": None,
                "passed": None,
                "classification": "insufficient_evidence",
                "explanation": (
                    "No meaningful concepts could be extracted "
                    "from the question."
                ),
                "matched_concepts": [],
                "missing_concepts": [],
                "evidence": None,
                "status": "insufficient_evidence"
            }

        matched = question_words.intersection(response_words)
        missing = question_words - matched

        score = round(
            len(matched) / len(question_words) * 100, 2
        )

        if score >= 80:
            classification = "fully_relevant"
        elif score >= 40:
            classification = "partially_relevant"
        elif score > 0:
            classification = "weakly_relevant"
        else:
            classification = "unrelated"

        explanation = (
            f"The response matched {len(matched)} out of "
            f"{len(question_words)} meaningful question concepts "
            f"({score}%). "
        )

        if matched:
            explanation += (
                "Matched concepts: "
                + ", ".join(sorted(matched)) + ". "
            )

        if missing:
            explanation += (
                "Missing concepts: "
                + ", ".join(sorted(missing)) + "."
            )

        return {
            "judge_name": "Relevance Judge",
            "score": score,
            "passed": score >= self.passing_threshold,
            "classification": classification,
            "explanation": explanation,
            "matched_concepts": sorted(matched),
            "missing_concepts": sorted(missing),
            "evidence": (
                f"Matched {len(matched)} out of "
                f"{len(question_words)} question concepts."
            ),
            "status": "evaluated"
        }
