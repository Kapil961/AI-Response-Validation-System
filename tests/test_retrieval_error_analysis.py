
import pytest

from evaluation.retrieval_error_analysis import RetrievalErrorAnalyzer


def make_result(title, paragraph_index):
    return {
        "metadata": {
            "title": title,
            "paragraph_index": paragraph_index,
        }
    }


@pytest.fixture
def analyzer():
    return RetrievalErrorAnalyzer(top_k=5)


@pytest.fixture
def evaluation_result():
    return {
        "question": "Which team represented the AFC?",
        "expected_title": "Super_Bowl_50",
        "expected_paragraph_index": 0,
        "results": [
            make_result("Super_Bowl_50", 0),
            make_result("Other_Article", 1),
            make_result("Another_Article", 0),
        ],
    }


def test_rank_one_success(analyzer, evaluation_result):
    result = analyzer.analyze_question(evaluation_result)

    assert result["category"] == "rank_1_success"
    assert result["first_relevant_rank"] == 1
    assert result["relevant_ranks"] == [1]


def test_lower_rank_success(analyzer, evaluation_result):
    evaluation_result["results"] = [
        make_result("Other_Article", 0),
        make_result("Super_Bowl_50", 0),
    ]

    result = analyzer.analyze_question(evaluation_result)

    assert result["category"] == "lower_rank_success"
    assert result["first_relevant_rank"] == 2


def test_retrieval_failure(analyzer, evaluation_result):
    evaluation_result["results"] = [
        make_result("Other_Article", 0),
        make_result("Another_Article", 1),
    ]

    result = analyzer.analyze_question(evaluation_result)

    assert result["category"] == "retrieval_failure"
    assert result["first_relevant_rank"] is None


def test_analyzer_counts_multiple_relevant_ranks(
    analyzer,
    evaluation_result,
):
    evaluation_result["results"] = [
        make_result("Super_Bowl_50", 0),
        make_result("Other_Article", 1),
        make_result("Super_Bowl_50", 0),
    ]

    result = analyzer.analyze_question(evaluation_result)

    assert result["relevant_ranks"] == [1, 3]
    assert result["first_relevant_rank"] == 1


def test_dataset_summary(analyzer):
    results = [
        {
            "question": "Question 1",
            "expected_title": "Article_A",
            "expected_paragraph_index": 0,
            "results": [
                make_result("Article_A", 0),
            ],
        },
        {
            "question": "Question 2",
            "expected_title": "Article_B",
            "expected_paragraph_index": 1,
            "results": [
                make_result("Other", 0),
                make_result("Article_B", 1),
            ],
        },
        {
            "question": "Question 3",
            "expected_title": "Article_C",
            "expected_paragraph_index": 0,
            "results": [
                make_result("Other", 0),
            ],
        },
    ]

    summary = analyzer.analyze_dataset(results)

    assert summary["total_questions"] == 3
    assert summary["rank_1_success"] == 1
    assert summary["lower_rank_success"] == 1
    assert summary["retrieval_failure"] == 1

    assert summary["rank_1_success_rate"] == pytest.approx(1 / 3)
    assert summary["lower_rank_success_rate"] == pytest.approx(1 / 3)
    assert summary["retrieval_failure_rate"] == pytest.approx(1 / 3)


def test_empty_dataset_rejected(analyzer):
    with pytest.raises(ValueError, match="cannot be empty"):
        analyzer.analyze_dataset([])


def test_invalid_top_k_rejected():
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        RetrievalErrorAnalyzer(top_k=0)


def test_missing_expected_title_raises_error(analyzer):
    result = {
        "question": "Test question",
        "expected_paragraph_index": 0,
        "results": [],
    }

    with pytest.raises(KeyError):
        analyzer.analyze_question(result)


def test_article_level_statistics(analyzer):
    results = [
        {
            "question": "Question 1",
            "expected_title": "Article_A",
            "expected_paragraph_index": 0,
            "results": [make_result("Article_A", 0)],
        },
        {
            "question": "Question 2",
            "expected_title": "Article_A",
            "expected_paragraph_index": 1,
            "results": [make_result("Other", 0)],
        },
        {
            "question": "Question 3",
            "expected_title": "Article_B",
            "expected_paragraph_index": 0,
            "results": [make_result("Other", 0)],
        },
    ]

    summary = analyzer.analyze_dataset(results)

    assert summary["total_articles"] == 2

    article_a = next(
        article
        for article in summary["article_statistics"]
        if article["title"] == "Article_A"
    )

    assert article_a["total_questions"] == 2
    assert article_a["rank_1_success"] == 1
    assert article_a["retrieval_failure"] == 1
    assert article_a["retrieval_failure_rate"] == 0.5


def test_articles_sorted_by_failure_count(analyzer):
    results = [
        {
            "question": "Question 1",
            "expected_title": "Article_A",
            "expected_paragraph_index": 0,
            "results": [make_result("Other", 0)],
        },
        {
            "question": "Question 2",
            "expected_title": "Article_A",
            "expected_paragraph_index": 1,
            "results": [make_result("Other", 0)],
        },
        {
            "question": "Question 3",
            "expected_title": "Article_B",
            "expected_paragraph_index": 0,
            "results": [make_result("Article_B", 0)],
        },
    ]

    summary = analyzer.analyze_dataset(results)

    articles = summary["article_statistics"]

    assert articles[0]["title"] == "Article_A"
    assert articles[0]["retrieval_failure"] == 2
