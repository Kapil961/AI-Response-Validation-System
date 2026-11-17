
from typing import Optional
from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    judge_name: str
    score: Optional[float] = Field(default=None, ge=0, le=100)
    passed: Optional[bool] = None
    explanation: str
    evidence: Optional[str] = None
    status: str = "evaluated"


class EvaluationResult(BaseModel):
    submission_id: str
    overall_score: float = Field(ge=0, le=100)

    relevance: JudgeResult
    accuracy: JudgeResult
    faithfulness: JudgeResult
    completeness: JudgeResult

    verdict: str
    summary: str
