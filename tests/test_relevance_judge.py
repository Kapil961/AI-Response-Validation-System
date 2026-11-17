
from agents.judges.relevance_judge import RelevanceJudge


def test_relevant_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning?",
        "Machine learning is a branch of artificial intelligence "
        "that allows computers to learn from data."
    )

    assert result["score"] > 0
    assert result["passed"] is True


def test_irrelevant_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning?",
        "The capital of France is Paris."
    )

    assert result["score"] == 0
    assert result["passed"] is False
