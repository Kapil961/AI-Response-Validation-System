
import re


class RelevanceJudge:

    def __init__(self):
        self.stop_words = {
            "what", "is", "are", "the", "a", "an",
            "of", "in", "on", "to", "for", "and",
            "how", "why", "when", "where", "does",
            "do", "can", "could", "would", "should"
        }

    def tokenize(self, text: str) -> set:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        return {
            word for word in words
            if word not in self.stop_words
        }

    def evaluate(self, question: str, response: str) -> dict:

        question_words = self.tokenize(question)
        response_words = self.tokenize(response)

        if not question_words:
            score = 0.0
        else:
            matched_words = question_words.intersection(response_words)

            score = (
                len(matched_words) / len(question_words)
            ) * 100

        score = round(score, 2)

        return {
            "judge_name": "Relevance Judge",
            "score": score,
            "passed": score >= 50,
            "explanation": (
                "The score measures the proportion of meaningful "
                "question words found in the response. "
                "This is a lexical-overlap baseline, not a semantic "
                "understanding score."
            ),
            "evidence": None
        }
