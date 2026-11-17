from agents.verdict_agent import VerdictAgent
from backend.models.evaluation import JudgeResult


def make_results(score):
    return {
        name: JudgeResult(
            judge_name=name,
            score=score,
            passed=score >= 70,
            explanation="Test evaluation",
        )
        for name in [
            "relevance",
            "accuracy",
            "faithfulness",
            "completeness",
        ]
    }


def test_verdict_agent_pass():
    agent = VerdictAgent()
    result = agent.evaluate(make_results(90))

    assert result["overall_score"] == 90
    assert result["verdict"] == "PASS"


def test_verdict_agent_review():
    agent = VerdictAgent()
    result = agent.evaluate(make_results(70))

    assert result["overall_score"] == 70
    assert result["verdict"] == "REVIEW"


def test_verdict_agent_fail():
    agent = VerdictAgent()
    result = agent.evaluate(make_results(40))

    assert result["overall_score"] == 40
    assert result["verdict"] == "FAIL"


def test_verdict_agent_missing_evidence():
    agent = VerdictAgent()
    results = make_results(90)

    results["accuracy"] = JudgeResult(
        judge_name="Accuracy Judge",
        score=None,
        passed=None,
        explanation="Reference answer not provided.",
        status="insufficient_evidence",
    )

    result = agent.evaluate(results)

    assert result["overall_score"] == 90
    assert result["verdict"] == "PASS"
    assert "accuracy" in result["summary"]


def test_verdict_agent_no_evidence():
    agent = VerdictAgent()

    results = {
        name: JudgeResult(
            judge_name=name,
            score=None,
            passed=None,
            explanation="Evidence unavailable.",
            status="insufficient_evidence",
        )
        for name in [
            "relevance",
            "accuracy",
            "faithfulness",
            "completeness",
        ]
    }

    result = agent.evaluate(results)

    assert result["overall_score"] == 0
    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"
