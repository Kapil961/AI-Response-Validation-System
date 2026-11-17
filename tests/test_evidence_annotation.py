import pytest
from pydantic import ValidationError

from evaluation.evidence_annotation import EvidenceAnnotation


def test_valid_evidence_annotation():
    annotation = EvidenceAnnotation(
        question="When did Martin Luther die?",
        gold_answer=["18 February 1546"],
        evidence=[
            "He died at 2:45 a.m. on 18 February 1546."
        ],
        label="supported",
        explanation="The passage explicitly states his date of death.",
    )

    assert annotation.label == "supported"
    assert annotation.reviewer == "human"


@pytest.mark.parametrize(
    "label",
    [
        "supported",
        "partially_supported",
        "unsupported",
        "insufficient_evidence",
    ],
)
def test_all_valid_evidence_labels(label):
    annotation = EvidenceAnnotation(
        question="Test question",
        gold_answer=["Test answer"],
        evidence=["Test evidence"],
        label=label,
        explanation="Test explanation",
    )

    assert annotation.label == label


def test_invalid_evidence_label():
    with pytest.raises(ValidationError):
        EvidenceAnnotation(
            question="Test question",
            gold_answer=["Test answer"],
            evidence=["Test evidence"],
            label="incorrect_label",
            explanation="Invalid label test",
        )


def test_empty_evidence_rejected():
    with pytest.raises(ValidationError):
        EvidenceAnnotation(
            question="Test question",
            gold_answer=["Test answer"],
            evidence=[],
            label="supported",
            explanation="Empty evidence test",
        )


def test_empty_question_rejected():
    with pytest.raises(ValidationError):
        EvidenceAnnotation(
            question="",
            gold_answer=["Test answer"],
            evidence=["Test evidence"],
            label="supported",
            explanation="Empty question test",
        )


def test_empty_gold_answer_rejected():
    with pytest.raises(ValidationError):
        EvidenceAnnotation(
            question="Test question",
            gold_answer=[],
            evidence=["Test evidence"],
            label="supported",
            explanation="Empty answer test",
        )


def test_empty_explanation_rejected():
    with pytest.raises(ValidationError):
        EvidenceAnnotation(
            question="Test question",
            gold_answer=["Test answer"],
            evidence=["Test evidence"],
            label="supported",
            explanation="",
        )
