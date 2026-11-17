# System Architecture

## 1. Overview

The AI Response Validation System evaluates AI-generated answers across four dimensions:

- Relevance
- Accuracy
- Evidence support (faithfulness)
- Completeness

The system uses a multi-judge architecture, a FastAPI backend, a Streamlit frontend, and a Retrieval-Augmented Generation (RAG) pipeline.

## 2. High-Level Architecture

```text
User
 |
 v
Streamlit Frontend
 |
 v
FastAPI Backend
 |
 v
Evaluation Orchestrator
 |
 +--> Relevance Judge
 |
 +--> Accuracy Judge
 |
 +--> Hallucination Detection Judge
 |
 +--> Completeness Judge
 |
 v
Verdict Agent
 |
 v
Structured Evaluation Result
## 3. Evaluation Agents

- Relevance Judge: Measures lexical overlap between the question and response.
- Accuracy Judge: Compares response tokens with an optional reference answer.
- Hallucination Detection Judge: Checks lexical evidence overlap with an optional source document.
- Completeness Judge: Estimates coverage of meaningful question concepts.

These are baseline lexical evaluation methods. They do not independently establish factual truth or semantic entailment.

## 4. Verdict Agent

The Verdict Agent combines available evaluation scores using weighted aggregation.

| Dimension | Weight |
|---|---:|
| Relevance | 20% |
| Accuracy | 30% |
| Faithfulness | 30% |
| Completeness | 20% |

Verdict thresholds:

- PASS: score >= 80
- REVIEW: score >= 60 and < 80
- FAIL: score < 60

## 5. RAG Pipeline

- Dataset Loader: Loads and standardizes datasets.
- Text Preprocessor: Normalizes and chunks source documents.
- Embedding Model: Generates 384-dimensional Sentence Transformer embeddings.
- NumPy Vector Store: Stores vectors and associated metadata.
- Retrieval Engine: Retrieves similar evidence chunks using cosine similarity.

## 6. Retrieval Evaluation

The retrieval evaluation module measures Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).

The error analyzer categorizes results into rank-1 success, lower-rank success, and retrieval failure.

## 7. Frontend

The Streamlit frontend accepts questions, AI responses, reference answers, and source documents. It communicates with the FastAPI backend and displays evaluation scores, explanations, and verdicts.

## 8. Technology Stack

- Python
- FastAPI
- Streamlit
- Pydantic
- Sentence Transformers
- PyTorch
- NumPy
- Pytest

## 9. Current Limitations

- Evaluation judges use lexical baselines rather than a fully implemented LLM judge.
- Lexical overlap cannot reliably identify semantic equivalence or contradictions.
- Evidence overlap does not prove factual support.
- Retrieval quality depends on the indexed corpus, embedding model, and configuration.
- The reported validation benchmark is an in-corpus evaluation, not an unseen-document generalization test.
