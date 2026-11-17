
from pydantic import BaseModel, Field, model_validator
from typing import Optional


class SubmissionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        description="The original question asked by the user."
    )

    ai_response: str = Field(
        ...,
        min_length=10,
        description="The AI-generated response to evaluate."
    )

    reference_answer: Optional[str] = Field(
        default=None,
        description="Optional trusted reference answer."
    )

    source_document: Optional[str] = Field(
        default=None,
        description="Optional source document or supporting context."
    )

    @model_validator(mode="after")
    def validate_submission(self):
        if not self.ai_response.strip():
            raise ValueError("AI response cannot be empty.")

        if not self.question.strip():
            raise ValueError("Question cannot be empty.")

        return self
