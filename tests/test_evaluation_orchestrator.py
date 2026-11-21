from backend.models.submission import SubmissionRequest
from backend.services.evaluation_orchestrator import EvaluationOrchestrator


def test_complete_evaluation():
    orchestrator = EvaluationOrchestrator()

    submission = SubmissionRequest(
        question="What is machine learning?",
        ai_response=(
            "Machine learning enables computers to learn patterns from data."
        ),
        reference_answer=(
            "Machine learning enables computers to learn patterns from data."
        ),
        source_document=(
            "Machine learning enables computers to learn patterns from data."
        ),
    )

    result = orchestrator.evaluate(submission)

    assert result.submission_id
    assert 0 <= result.overall_score <= 100
    assert result.relevance.score is not None
    assert result.accuracy.score is not None
    assert result.faithfulness.score is not None
    assert result.completeness.score is not None
    assert result.verdict == "PASS"


def test_evaluation_without_reference_and_source():
    orchestrator = EvaluationOrchestrator()

    submission = SubmissionRequest(
        question="What is machine learning?",
        ai_response=(
            "Machine learning enables computers to learn patterns from data."
        ),
    )

    result = orchestrator.evaluate(submission)

    assert result.accuracy.score is None
    assert result.faithfulness.score is None
    assert result.relevance.score is not None
    assert result.completeness.score is not None
    assert "accuracy" in result.summary
    assert "faithfulness" in result.summary
def test_relevance_classification_in_orchestrator():
    orchestrator = EvaluationOrchestrator()

    submission = SubmissionRequest(
        question=(
            "What is machine learning and artificial intelligence?"
        ),
        ai_response=(
            "Machine learning is a branch of artificial intelligence."
        ),
        reference_answer=(
            "Machine learning is a branch of artificial intelligence."
        ),
        source_document=(
            "Machine learning is a branch of artificial intelligence."
        ),
    )

    result = orchestrator.evaluate(submission)

    assert result.relevance.classification == "fully_relevant"
    assert result.relevance.score == 100.0
    assert "machine" in result.relevance.matched_concepts
    assert result.relevance.status == "evaluated"
class MockRetrievalEngine:

    def __init__(self, results):
        self.results = results
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        return self.results


def test_retrieval_used_when_source_document_missing():
    passage = (
        "Computer programs that learn and adapt are part "
        "of the emerging field of artificial intelligence "
        "and machine learning."
    )

    mock_engine = MockRetrievalEngine([
        {"text": passage, "similarity": 0.90}
    ])

    orchestrator = EvaluationOrchestrator(
        retrieval_engine=mock_engine
    )

    submission = SubmissionRequest(
        question="What is machine learning?",
        ai_response=passage,
    )

    result = orchestrator.evaluate(submission)

    assert mock_engine.calls == 1
    assert result.faithfulness.status == "evaluated"
    assert result.faithfulness.score == 100.0


def test_manual_source_document_takes_priority():
    mock_engine = MockRetrievalEngine([
        {
            "text": "Unrelated information about computer networks.",
            "similarity": 0.50
        }
    ])

    source = (
        "Machine learning enables computers to learn "
        "patterns from data."
    )

    orchestrator = EvaluationOrchestrator(
        retrieval_engine=mock_engine
    )

    submission = SubmissionRequest(
        question="What is machine learning?",
        ai_response=source,
        source_document=source,
    )

    result = orchestrator.evaluate(submission)

    assert mock_engine.calls == 0
    assert result.faithfulness.status == "evaluated"
    assert result.faithfulness.score == 100.0


def test_empty_retrieval_returns_insufficient_evidence():
    mock_engine = MockRetrievalEngine([])

    orchestrator = EvaluationOrchestrator(
        retrieval_engine=mock_engine
    )

    submission = SubmissionRequest(
        question="What is machine learning?",
        ai_response=(
            "Machine learning enables computers to learn patterns from data."
        ),
    )

    result = orchestrator.evaluate(submission)

    assert mock_engine.calls == 1
    assert result.faithfulness.status == "insufficient_evidence"
    assert result.faithfulness.score is None
