import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import json

from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.retrieval_error_analysis import RetrievalErrorAnalyzer
from rag.ingestion.dataset_loader import DatasetLoader
from rag.retrieval.retrieval_engine import RetrievalEngine


def main():
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = project_root / "datasets/raw/squad_dev.json"
    index_path = project_root / "rag/vector_store/squad_dev"
    report_path = project_root / "docs/reports/retrieval_error_analysis.json"

    print("Loading SQuAD validation questions...")

    question_records = DatasetLoader.load_squad_json(
    str(dataset_path)
)

    print(f"Loaded questions: {len(question_records)}")

    print("\nLoading retrieval engine...")

    retrieval_engine = RetrievalEngine(
        storage_path=index_path,
        top_k=5,
    )

    evaluator = RetrievalEvaluator(
        retrieval_engine=retrieval_engine,
        top_k=5,
    )

    print("\nStarting retrieval evaluation...")

    evaluation_report = evaluator.evaluate_dataset(question_records)

    print("\nStarting retrieval error analysis...")

    analyzer = RetrievalErrorAnalyzer(top_k=5)

    error_report = analyzer.analyze_dataset(
        evaluation_report["question_results"]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(error_report, file, indent=4, ensure_ascii=False)

    print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Total questions: {evaluation_report['total_questions']}")
    print(f"Recall@5: {evaluation_report['recall_at_k']:.4f}")
    print(f"Precision@5: {evaluation_report['precision_at_k']:.4f}")
    print(f"MRR: {evaluation_report['mrr']:.4f}")

    print("\n========== RETRIEVAL ERROR ANALYSIS ==========")
    print(f"Rank-1 success: {error_report['rank_1_success']}")
    print(f"Lower-rank success: {error_report['lower_rank_success']}")
    print(f"Retrieval failure: {error_report['retrieval_failure']}")

    print(
        f"Rank-1 success rate: "
        f"{error_report['rank_1_success_rate']:.2%}"
    )
    print(
        f"Lower-rank success rate: "
        f"{error_report['lower_rank_success_rate']:.2%}"
    )
    print(
        f"Retrieval failure rate: "
        f"{error_report['retrieval_failure_rate']:.2%}"
    )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
