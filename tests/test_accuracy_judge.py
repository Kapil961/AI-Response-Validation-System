
from agents.judges.accuracy_judge import AccuracyJudge


def test_matching_answer():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="Machine learning enables computers to learn from data.",
        reference_answer="Machine learning enables computers to learn from data."
    )

    assert result["score"] == 100
    assert result["passed"] is True
    assert result["classification"] == "correct"
    assert result["status"] == "evaluated"


def test_partially_correct_answer():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="Machine learning enables computers to learn.",
        reference_answer=(
            "Machine learning enables computers to learn "
            "patterns from data."
        )
    )

    assert 50 <= result["score"] < 85
    assert result["classification"] == "partially_correct"


def test_incorrect_answer():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="The capital of France is Paris.",
        reference_answer=(
            "Machine learning enables computers to learn from data."
        )
    )

    assert result["score"] < 50
    assert result["classification"] == "incorrect"
    assert result["passed"] is False


def test_missing_reference():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="Machine learning enables computers to learn from data."
    )

    assert result["score"] is None
    assert result["passed"] is None
    assert result["classification"] == "insufficient_evidence"
    assert result["status"] == "insufficient_evidence"


def test_accuracy_evidence_fields():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="Machine learning enables computers to learn.",
        reference_answer=(
            "Machine learning enables computers to learn patterns from data."
        )
    )

    assert "machine" in result["matched_concepts"]
    assert "learning" in result["matched_concepts"]
    assert "patterns" in result["missing_concepts"]
    assert result["explanation"]
    assert result["evidence"]


def test_empty_response():
    judge = AccuracyJudge()

    result = judge.evaluate(
        response="",
        reference_answer="Machine learning uses data."
    )

    assert result["score"] == 0.0
    assert result["classification"] == "incorrect"
    assert result["status"] == "evaluated"
