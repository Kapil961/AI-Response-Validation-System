
import csv
import json

import pytest

from rag.ingestion.dataset_loader import DatasetLoader


def test_normalize_record():
    record = DatasetLoader.normalize_record(
        dataset="truthfulqa",
        question="What is AI?",
        answer="Artificial intelligence.",
        context="AI is a field of computer science.",
    )

    assert record["dataset"] == "truthfulqa"
    assert record["question"] == "What is AI?"
    assert record["answer"] == "Artificial intelligence."
    assert record["context"] == "AI is a field of computer science."


def test_normalize_record_rejects_empty_question():
    with pytest.raises(ValueError):
        DatasetLoader.normalize_record(
            dataset="truthfulqa",
            question=" ",
            answer="Some answer",
        )


def test_load_truthfulqa_csv(tmp_path):
    file_path = tmp_path / "truthfulqa.csv"

    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Question",
                "Best Answer",
                "Category",
                "Type",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "Question": "What is AI?",
            "Best Answer": "AI is artificial intelligence.",
            "Category": "Technology",
            "Type": "Knowledge",
        })

    records = DatasetLoader.load_truthfulqa_csv(file_path)

    assert len(records) == 1
    assert records[0]["dataset"] == "truthfulqa"
    assert records[0]["question"] == "What is AI?"
    assert records[0]["answer"] == "AI is artificial intelligence."
    assert records[0]["metadata"]["category"] == "Technology"


def test_load_squad_json(tmp_path):
    file_path = tmp_path / "squad.json"

    sample_data = {
        "data": [
            {
                "title": "Artificial Intelligence",
                "paragraphs": [
                    {
                        "context": "AI enables machines to perform intelligent tasks.",
                        "qas": [
                            {
                                "id": "question-001",
                                "question": "What does AI enable machines to do?",
                                "answers": [
                                    {
                                        "text": "Perform intelligent tasks.",
                                        "answer_start": 35,
                                    }
                                ],
                                "is_impossible": False,
                            }
                        ],
                    }
                ],
            }
        ]
    }

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(sample_data, file)

    records = DatasetLoader.load_squad_json(file_path)

    assert len(records) == 1
    assert records[0]["dataset"] == "squad"
    assert records[0]["question"] == "What does AI enable machines to do?"
    assert records[0]["answer"] == "Perform intelligent tasks."
    assert records[0]["context"] == (
        "AI enables machines to perform intelligent tasks."
    )
    assert records[0]["metadata"]["title"] == "Artificial Intelligence"


def test_loader_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        DatasetLoader.load_squad_json("missing_squad.json")


def test_loader_rejects_invalid_squad_format(tmp_path):
    file_path = tmp_path / "invalid_squad.json"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump({"questions": []}, file)

    with pytest.raises(ValueError):
        DatasetLoader.load_squad_json(file_path)


def test_parse_semicolon_separated_answers():
    """Ensure semicolon-separated answers are parsed individually."""

    answers = DatasetLoader._parse_answers(
        "First answer; Second answer; Third answer"
    )

    assert answers == [
        "First answer",
        "Second answer",
        "Third answer",
    ]


def test_load_squad_documents_deduplicates_contexts(tmp_path):
    """Ensure duplicate paragraphs are stored only once."""

    dataset = {
        "data": [
            {
                "title": "Article One",
                "paragraphs": [
                    {"context": "This is a sample paragraph."},
                    {"context": "This is a sample paragraph."},
                    {"context": "This is another paragraph."},
                ],
            }
        ]
    }

    file_path = tmp_path / "sample_squad.json"
    file_path.write_text(json.dumps(dataset), encoding="utf-8")

    documents = DatasetLoader.load_squad_documents(
        file_path,
        split="train",
    )

    assert len(documents) == 2
    assert documents[0]["text"] == "This is a sample paragraph."
    assert documents[0]["metadata"]["split"] == "train"
    assert documents[0]["metadata"]["title"] == "Article One"
    assert documents[0]["metadata"]["source_file"] == "sample_squad.json"
    assert documents[0]["document_id"]
    assert documents[0]["document_id"] != documents[1]["document_id"]
