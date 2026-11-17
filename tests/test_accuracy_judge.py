
from agents.judges.accuracy_judge import AccuracyJudge


def test_matching_answer():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="Machine learning enables computers to learn from data.",
        reference_answer="Machine learning enables computers to learn from data."
    )

    assert result["score"] == 100
    assert result["passed"] is True
    assert result["status"] == "evaluated"


def test_different_answer():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="The capital of France is Paris.",
        reference_answer="Machine learning enables computers to learn from data."
    )

    assert result["score"] == 0
    assert result["passed"] is False


def test_missing_reference():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="Machine learning enables computers to learn from data."
    )

    assert result["score"] is None
    assert result["passed"] is None
    assert result["status"] == "insufficient_evidence"
