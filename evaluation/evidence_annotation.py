from typing import Literal

from pydantic import BaseModel, Field


EvidenceLabel = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "insufficient_evidence",
]


class EvidenceAnnotation(BaseModel):
    """
    Human-reviewed assessment of retrieved evidence.
    """

    question: str = Field(min_length=1)

    gold_answer: list[str] = Field(
        min_length=1
    )

    evidence: list[str] = Field(
        min_length=1
    )

    label: EvidenceLabel

    explanation: str = Field(
        min_length=1
    )

    reviewer: str = Field(
        default="human"
    )
