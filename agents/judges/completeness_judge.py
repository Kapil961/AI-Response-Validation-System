
import re


class CompletenessJudge:

    def __init__(self, completeness_threshold: float = 70.0):
        self.completeness_threshold = completeness_threshold


        self.stop_words = {
    "what", "is", "are", "the", "a", "an",
    "of", "in", "on", "to", "for", "and",
    "how", "why", "when", "where", "does",
    "do", "can", "could", "would", "should",
    "explain", "describe", "discuss", "define",
    "tell", "me", "about", "give", "provide",
    "it", "this", "that", "these", "those",
    "he", "she", "they", "we", "you", "i",
    "be", "been", "being", "was", "were",
    "has", "have", "had", "your", "my"
}


    def tokenize(self, text: str) -> set:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return {
            word for word in words
            if word not in self.stop_words
        }

    def evaluate(self, question: str, response: str) -> dict:

        expected_concepts = self.tokenize(question)
        response_concepts = self.tokenize(response)

        if not expected_concepts:
            return {
                "judge_name": "Completeness Judge",
                "score": None,
                "passed": None,
                "status": "insufficient_evidence",
                "explanation": (
                    "No meaningful concepts could be extracted "
                    "from the question."
                ),
                "missing_concepts": [],
                "covered_concepts": []
            }

        covered_concepts = expected_concepts.intersection(
            response_concepts
        )

        missing_concepts = expected_concepts - covered_concepts

        score = round(
            len(covered_concepts) / len(expected_concepts) * 100,
            2
        )

        return {
            "judge_name": "Completeness Judge",
            "score": score,
            "passed": score >= self.completeness_threshold,
            "status": "evaluated",
            "explanation": (
                "Completeness is estimated using the proportion "
                "of meaningful question concepts appearing in "
                "the response. This is a lexical baseline."
            ),
            "missing_concepts": sorted(missing_concepts),
            "covered_concepts": sorted(covered_concepts)
        }
