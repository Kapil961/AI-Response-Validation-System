from fastapi import FastAPI, HTTPException

from backend.models.submission import SubmissionRequest
from backend.services.evaluation_orchestrator import EvaluationOrchestrator


app = FastAPI(
    title="AI Response Validation System",
    description=(
        "An AI-powered system for evaluating response quality "
        "and detecting hallucinations."
    ),
    version="1.0.0",
)

# Initialize the evaluation orchestrator
orchestrator = EvaluationOrchestrator()


@app.get("/")
def root():
    return {
        "message": "AI Response Validation System API is running!",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/submissions")
def create_submission(submission: SubmissionRequest):
    try:
        evaluation_result = orchestrator.evaluate(submission)

        return {
            "message": "Evaluation completed successfully!",
            "result": evaluation_result.model_dump(),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(error)}",
        )
