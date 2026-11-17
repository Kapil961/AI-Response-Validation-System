
from agents.judges.completeness_judge import CompletenessJudge


def test_complete_response():
    judge = CompletenessJudge()

    result = judge.evaluate(
        question="Explain machine learning, its types, and applications.",
        response=(
            "Machine learning is a field of AI. "
            "Its types include supervised and unsupervised learning. "
            "Applications include healthcare and finance."
        )
    )

    assert result["score"] == 100
    assert result["passed"] is True
    assert result["status"] == "evaluated"


def test_incomplete_response():
    judge = CompletenessJudge()

    result = judge.evaluate(
        question="Explain machine learning, its types, and applications.",
        response=(
            "Machine learning enables computers "
            "to learn patterns from data."
        )
    )

    assert result["score"] < 70
    assert result["passed"] is False
    assert "types" in result["missing_concepts"]
    assert "applications" in result["missing_concepts"]


def test_question_without_meaningful_concepts():
    judge = CompletenessJudge()

    result = judge.evaluate(
        question="What is it?",
        response="It is a system."
    )

    assert result["score"] is None
    assert result["passed"] is None
    assert result["status"] == "insufficient_evidence"
