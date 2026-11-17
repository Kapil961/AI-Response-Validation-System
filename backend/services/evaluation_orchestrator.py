from uuid import uuid4

from backend.models.evaluation import EvaluationResult, JudgeResult
from backend.models.submission import SubmissionRequest

from agents.judges.relevance_judge import RelevanceJudge
from agents.judges.accuracy_judge import AccuracyJudge
from agents.judges.hallucination_judge import HallucinationJudge
from agents.judges.completeness_judge import CompletenessJudge
from agents.verdict_agent import VerdictAgent


class EvaluationOrchestrator:
    def __init__(self):
        self.relevance_judge = RelevanceJudge()
        self.accuracy_judge = AccuracyJudge()
        self.hallucination_judge = HallucinationJudge()
        self.completeness_judge = CompletenessJudge()
        self.verdict_agent = VerdictAgent()

    def evaluate(self, submission: SubmissionRequest) -> EvaluationResult:
        relevance = JudgeResult(
            **self.relevance_judge.evaluate(
                submission.question, submission.ai_response
            )
        )

        accuracy = JudgeResult(
            **self.accuracy_judge.evaluate(
                response=submission.ai_response,
                reference_answer=submission.reference_answer,
            )
        )

        faithfulness_data = self.hallucination_judge.evaluate(
            response=submission.ai_response,
            source_document=submission.source_document,
        )

        claim_evidence = "; ".join(
            f'{claim["classification"]}: {claim["claim"]}'
            for claim in faithfulness_data.get("claims", [])
        ) or None

        faithfulness = JudgeResult(
            judge_name=faithfulness_data["judge_name"],
            score=faithfulness_data["score"],
            passed=faithfulness_data["passed"],
            explanation=faithfulness_data["explanation"],
            evidence=claim_evidence,
            status=faithfulness_data["status"],
        )

        completeness = JudgeResult(
            **self.completeness_judge.evaluate(
                question=submission.question,
                response=submission.ai_response,
            )
        )

        judge_results = {
            "relevance": relevance,
            "accuracy": accuracy,
            "faithfulness": faithfulness,
            "completeness": completeness,
        }

        verdict_result = self.verdict_agent.evaluate(judge_results)

        return EvaluationResult(
            submission_id=str(uuid4()),
            overall_score=verdict_result["overall_score"],
            relevance=relevance,
            accuracy=accuracy,
            faithfulness=faithfulness,
            completeness=completeness,
            verdict=verdict_result["verdict"],
            summary=verdict_result["summary"],
        )
