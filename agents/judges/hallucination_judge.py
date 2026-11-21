
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
        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def has_negation(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(not|never|no)\b",
                text.lower()
            )
        )

    def is_direct_contradiction(
        self,
        claim: str,
        source_document: str
    ) -> bool:

        claim_negated = self.has_negation(claim)

        ignored_words = {
            "not", "never", "no",
            "do", "does", "did",
            "can", "could", "will",
            "would", "should"
        }

        def normalize(text):
            terms = self.tokenize(text) - ignored_words

            return {
                word[:-1]
                if word.endswith("s") and len(word) > 4
                else word
                for word in terms
            }

        claim_terms = normalize(claim)

        if not claim_terms:
            return False

        for source_sentence in self.split_sentences(source_document):

            source_negated = self.has_negation(source_sentence)

            if claim_negated == source_negated:
                continue

            source_terms = normalize(source_sentence)

            if not source_terms:
                continue

            overlap = len(claim_terms & source_terms)

            similarity = overlap / min(
                len(claim_terms),
                len(source_terms)
            )

            if similarity >= 0.80:
                return True

        return False

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
                "explanation": (
                    "The source document contains no usable evidence."
                ),
                "claims": []
            }

        sentences = self.split_sentences(response)
        claims = []

        for sentence in sentences:
            claim_words = self.tokenize(sentence)

            if not claim_words:
                continue

            if self.is_direct_contradiction(
                sentence,
                source_document
            ):
                claims.append({
                    "claim": sentence,
                    "support_score": 0.0,
                    "classification": "contradictory",
                    "matched_evidence_terms": []
                })
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
                "explanation": (
                    "No evaluable claims were found in the response."
                ),
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
                "sentences and the source document, with a rule-based "
                "direct-negation check for potential contradictions. "
                "Other potentially unsupported claims require further "
                "verification and are not proven false."
            ),
            "claims": claims
        }
