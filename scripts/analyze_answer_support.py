import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.ingestion.dataset_loader import DatasetLoader
from rag.retrieval.retrieval_engine import RetrievalEngine


def normalize(text):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower()
    ).strip()


def main():
    report_path = (
        PROJECT_ROOT
        / "docs/reports/retrieval_error_analysis.json"
    )

    dataset_path = (
        PROJECT_ROOT
        / "datasets/raw/squad_dev.json"
    )

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    records = DatasetLoader.load_squad_json(
        str(dataset_path)
    )

    answer_map = {
        (
            record["question"],
            record["metadata"]["title"],
            record["metadata"]["paragraph_index"],
        ): record["answer"]
        for record in records
    }

    engine = RetrievalEngine(
        storage_path=str(
            PROJECT_ROOT / "rag/vector_store/squad_dev"
        ),
        top_k=5,
    )

    failures = [
        question
        for question in report["questions"]
        if question["category"] == "retrieval_failure"
        and question["expected_title"] == "Martin_Luther"
    ]

    results_report = []

    for failure in failures:
        key = (
            failure["question"],
            failure["expected_title"],
            failure["expected_paragraph_index"],
        )

        answer = answer_map.get(key)

        if answer is None:
            continue

        answers = answer if isinstance(answer, list) else [answer]

        retrieved = engine.retrieve(failure["question"])

        matched_results = []

        for rank, result in enumerate(retrieved, start=1):
            text = result["text"]

            matched_answers = [
                str(candidate)
                for candidate in answers
                if str(candidate).strip()
                and normalize(str(candidate)) in normalize(text)
            ]

            if matched_answers:
                matched_results.append({
                    "rank": rank,
                    "paragraph_index": result["metadata"].get(
                        "paragraph_index"
                    ),
                    "similarity": float(result["similarity"]),
                    "matched_answers": matched_answers,
                    "evidence": text,
                })

        results_report.append({
            "question": failure["question"],
            "expected_title": failure["expected_title"],
            "expected_paragraph_index": failure[
                "expected_paragraph_index"
            ],
            "gold_answer": answers,
            "answer_string_found": bool(matched_results),
            "matched_results": matched_results,
        })

    answer_found = sum(
        item["answer_string_found"]
        for item in results_report
    )

    output = {
        "dataset": "SQuAD validation",
        "article": "Martin_Luther",
        "top_k": 5,
        "total_exact_paragraph_failures": len(failures),
        "questions_analyzed": len(results_report),
        "answer_string_matches": answer_found,
        "answer_string_not_found": (
            len(results_report) - answer_found
        ),
        "answer_string_match_rate": (
            answer_found / len(results_report)
            if results_report else 0.0
        ),
        "questions": results_report,
    }

    output_path = (
        PROJECT_ROOT
        / "docs/reports/martin_luther_answer_support.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Questions analyzed:", len(results_report))
    print("Answer-string matches:", answer_found)
    print(
        "Answer-string match rate:",
        f"{output['answer_string_match_rate']:.2%}",
    )
    print("Report saved:", output_path)


if __name__ == "__main__":
    main()
