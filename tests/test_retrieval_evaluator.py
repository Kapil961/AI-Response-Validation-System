
import pytest

from evaluation.retrieval_evaluator import RetrievalEvaluator


class MockRetrievalEngine:
    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_top_k = None

    def retrieve(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k
        return self.results[:top_k]


@pytest.fixture
def question_record():
    return {
        "question": "Which NFL team represented the AFC at Super Bowl 50?",
        "answer": "Denver Broncos",
        "context": "The Denver Broncos represented the AFC.",
        "metadata": {
            "title": "Super_Bowl_50",
            "paragraph_index": 0,
        },
    }


def make_result(title, paragraph_index, similarity=0.8):
    return {
        "chunk_id": f"chunk_{title}_{paragraph_index}",
        "document_id": "test_document",
        "text": "Test source paragraph.",
        "metadata": {
            "title": title,
            "paragraph_index": paragraph_index,
        },
        "similarity": similarity,
    }


def test_evaluator_identifies_relevant_source(question_record):
    engine = MockRetrievalEngine([
        make_result("Super_Bowl_50", 0),
        make_result("Other_Article", 1),
    ])

    evaluator = RetrievalEvaluator(engine, top_k=2)

    result = evaluator.evaluate_question(question_record)

    assert result["relevant_count"] == 1
    assert result["recall_at_k"] == 1.0
    assert result["precision_at_k"] == 0.5
    assert result["reciprocal_rank"] == 1.0


def test_evaluator_calculates_reciprocal_rank(question_record):
    engine = MockRetrievalEngine([
        make_result("Other_Article", 0),
        make_result("Super_Bowl_50", 0),
        make_result("Another_Article", 0),
    ])

    evaluator = RetrievalEvaluator(engine, top_k=3)

    result = evaluator.evaluate_question(question_record)

    assert result["reciprocal_rank"] == 0.5
    assert result["recall_at_k"] == 1.0


def test_evaluator_returns_zero_when_source_missing(question_record):
    engine = MockRetrievalEngine([
        make_result("Other_Article", 0),
        make_result("Another_Article", 1),
    ])

    evaluator = RetrievalEvaluator(engine, top_k=2)

    result = evaluator.evaluate_question(question_record)

    assert result["relevant_count"] == 0
    assert result["recall_at_k"] == 0.0
    assert result["precision_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_evaluator_matches_paragraph_index(question_record):
    engine = MockRetrievalEngine([
        make_result("Super_Bowl_50", 1),
        make_result("Super_Bowl_50", 0),
    ])

    evaluator = RetrievalEvaluator(engine, top_k=2)

    result = evaluator.evaluate_question(question_record)

    assert result["relevant_count"] == 1
    assert result["precision_at_k"] == 0.5
    assert result["reciprocal_rank"] == 0.5


def test_evaluator_rejects_invalid_question(question_record):
    engine = MockRetrievalEngine([])
    evaluator = RetrievalEvaluator(engine)

    question_record["question"] = "   "

    with pytest.raises(ValueError, match="non-empty string"):
        evaluator.evaluate_question(question_record)


def test_evaluator_rejects_missing_title(question_record):
    engine = MockRetrievalEngine([])
    evaluator = RetrievalEvaluator(engine)

    question_record["metadata"].pop("title")

    with pytest.raises(ValueError, match="title is missing"):
        evaluator.evaluate_question(question_record)


def test_evaluator_rejects_missing_paragraph_index(question_record):
    engine = MockRetrievalEngine([])
    evaluator = RetrievalEvaluator(engine)

    question_record["metadata"].pop("paragraph_index")

    with pytest.raises(ValueError, match="paragraph index is missing"):
        evaluator.evaluate_question(question_record)


def test_evaluator_rejects_empty_dataset():
    engine = MockRetrievalEngine([])
    evaluator = RetrievalEvaluator(engine)

    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate_dataset([])


def test_evaluator_rejects_invalid_top_k():
    engine = MockRetrievalEngine([])

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        RetrievalEvaluator(engine, top_k=0)


def test_evaluator_calculates_dataset_metrics(question_record):
    engine = MockRetrievalEngine([
        make_result("Super_Bowl_50", 0),
        make_result("Other_Article", 1),
    ])

    evaluator = RetrievalEvaluator(engine, top_k=2)

    results = evaluator.evaluate_dataset(
        [question_record, question_record],
    )

    assert results["total_questions"] == 2
    assert results["top_k"] == 2
    assert results["recall_at_k"] == 1.0
    assert results["precision_at_k"] == 0.5
    assert results["mrr"] == 1.0


def test_evaluator_respects_dataset_limit(question_record):
    engine = MockRetrievalEngine([
        make_result("Super_Bowl_50", 0),
    ])

    evaluator = RetrievalEvaluator(engine, top_k=1)

    results = evaluator.evaluate_dataset(
        [question_record] * 5,
        limit=2,
    )

    assert results["total_questions"] == 2


def test_evaluator_rejects_invalid_dataset_limit(question_record):
    engine = MockRetrievalEngine([])
    evaluator = RetrievalEvaluator(engine)

    with pytest.raises(ValueError, match="limit must be greater than zero"):
        evaluator.evaluate_dataset([question_record], limit=0)
