
from agents.judges.relevance_judge import RelevanceJudge


def test_fully_relevant_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning and artificial intelligence?",
        "Machine learning is a branch of artificial intelligence."
    )

    assert result["classification"] == "fully_relevant"
    assert result["score"] == 100.0
    assert result["passed"] is True


def test_partially_relevant_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning and artificial intelligence?",
        "Machine learning uses data."
    )

    assert result["classification"] == "partially_relevant"
    assert result["score"] == 50.0


def test_weakly_relevant_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning and artificial intelligence?",
        "Machine."
    )

    assert result["classification"] == "weakly_relevant"
    assert result["score"] == 25.0


def test_unrelated_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning?",
        "The capital of France is Paris."
    )

    assert result["classification"] == "unrelated"
    assert result["score"] == 0.0
    assert result["passed"] is False


def test_relevance_evidence_fields():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning and artificial intelligence?",
        "Machine learning uses data."
    )

    assert "machine" in result["matched_concepts"]
    assert "learning" in result["matched_concepts"]
    assert "artificial" in result["missing_concepts"]
    assert result["explanation"]
    assert result["evidence"]


def test_empty_question_returns_insufficient_evidence():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is?",
        "Machine learning uses data."
    )

    assert result["classification"] == "insufficient_evidence"
    assert result["score"] is None
    assert result["status"] == "insufficient_evidence"
