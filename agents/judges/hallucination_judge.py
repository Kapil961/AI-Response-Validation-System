
import re
from typing import Optional


class HallucinationJudge:

    def __init__(self, support_threshold: float = 0.70):
        self.support_threshold = support_threshold

        self.stop_words = {
            "the", "is", "are", "was", "were", "a", "an",
            "and", "or", "but", "of", "in", "on", "at",
            "to", "for", "from", "by", "with", "as",
            "it", "this", "that", "these", "those",
            "he", "she", "they", "we", "you", "i",
            "has", "have", "had", "be", "been", "being"
        }

    def tokenize(self, text: str) -> set:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return {
            word for word in words
            if word not in self.stop_words
        }

    def split_sentences(self, text: str) -> list:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def evaluate(
        self,
        response: str,
        source_document: Optional[str] = None
    ) -> dict:

        if not source_document or not source_document.strip():
            return {
                "judge_name": "Hallucination Detection Agent",
                "score": None,
                "passed": None,
                "status": "insufficient_evidence",
                "explanation": (
                    "No source document was provided. "
                    "Claims cannot be checked for evidence support."
                ),
                "claims": []
            }

        source_words = self.tokenize(source_document)

        if not source_words:
            return {
                "judge_name": "Hallucination Detection Agent",
                "score": None,
                "passed": None,
                "status": "insufficient_evidence",
                "explanation": "The source document contains no usable evidence.",
                "claims": []
            }

        sentences = self.split_sentences(response)
        claims = []

        for sentence in sentences:
            claim_words = self.tokenize(sentence)

            if not claim_words:
                continue

            matched_words = claim_words.intersection(source_words)

            support_score = (
                len(matched_words) / len(claim_words)
            ) * 100

            support_score = round(support_score, 2)

            if support_score >= self.support_threshold * 100:
                classification = "supported"
            else:
                classification = "potentially_unsupported"

            claims.append({
                "claim": sentence,
                "support_score": support_score,
                "classification": classification,
                "matched_evidence_terms": sorted(matched_words)
            })

        if not claims:
            return {
                "judge_name": "Hallucination Detection Agent",
                "score": None,
                "passed": None,
                "status": "insufficient_evidence",
                "explanation": "No evaluable claims were found in the response.",
                "claims": []
            }

        supported_count = sum(
            claim["classification"] == "supported"
            for claim in claims
        )

        overall_score = round(
            (supported_count / len(claims)) * 100, 2
        )

        return {
            "judge_name": "Hallucination Detection Agent",
            "score": overall_score,
            "passed": overall_score >= 70,
            "status": "evaluated",
            "explanation": (
                "This baseline checks lexical overlap between response "
                "sentences and the source document. Potentially unsupported "
                "claims require further verification; they are not proven false."
            ),
            "claims": claims
        }
