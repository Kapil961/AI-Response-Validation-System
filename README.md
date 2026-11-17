# AI Response Validation System with Hallucination Detection Assistance

## Project Overview

An AI-powered response validation system designed to evaluate AI-generated responses across multiple quality dimensions, including relevance, factual accuracy, completeness, and evidence support.

The system combines a multi-judge evaluation architecture with Retrieval-Augmented Generation (RAG) to retrieve relevant reference evidence and assist in identifying potentially unsupported claims.

Developed as part of the **Infosys Springboard Virtual Internship**.

## Key Features

- AI-generated response evaluation
- Relevance, accuracy, and completeness assessment
- Hallucination detection assistance through evidence-support analysis
- Retrieval-Augmented Generation (RAG) pipeline
- Semantic search using Sentence Transformers
- NumPy-based vector storage and similarity search
- Multi-judge evaluation architecture
- Evidence retrieval and retrieval-quality evaluation
- Retrieval error analysis and reporting
- FastAPI backend
- Streamlit interactive frontend
- Automated test coverage

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Data Validation | Pydantic |
| Embedding Model | Sentence Transformers |
| Deep Learning Framework | PyTorch |
| Vector Storage | NumPy |
| Similarity Search | Cosine Similarity |
| Testing | Pytest |
| Retrieval Evaluation | Recall@K, Precision@K, MRR |

## System Architecture

The project is organized into the following major components:

- **Backend:** Handles API requests, input validation, and evaluation orchestration.
- **Evaluation Agents:** Evaluate relevance, accuracy, completeness, and evidence support.
- **RAG Pipeline:** Loads datasets, preprocesses documents, generates embeddings, and indexes document chunks.
- **Retrieval Engine:** Retrieves relevant evidence using semantic similarity.
- **Evaluation Module:** Measures retrieval performance and analyzes retrieval failures.
- **Frontend:** Provides an interactive interface for submitting AI responses and viewing evaluation results.

Detailed architecture documentation is available in `docs/architecture/system_architecture.md`.

## Project Structure

```text
ai-response-validation-system/
├── agents/
├── backend/
├── datasets/
├── docs/
├── evaluation/
├── frontend/
├── rag/
├── scripts/
├── tests/
├── requirements.txt
├── requirements-frontend.txt
├── requirements-rag.txt
└── README.md
```

## Installation and Setup

The project uses separate Python environments for the backend, frontend, and embedding pipeline.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ai-response-validation-system
```

### 2. Backend Environment

Create and activate the backend environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Frontend Environment

Create and activate the frontend environment:

```bash
python3.11 -m venv venv-py311
source venv-py311/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements-frontend.txt
```

### 4. RAG and Embeddings Environment

Create and activate the RAG environment:

```bash
python3.11 -m venv venv-embeddings
source venv-embeddings/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements-rag.txt
```

The embedding model is downloaded automatically when it is first loaded.

## Running the Application

### Start the Backend

In the first terminal:

```bash
source venv/bin/activate
uvicorn backend.main:app --reload
```

Backend API:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

### Start the Frontend

In a second terminal:

```bash
source venv-py311/bin/activate
streamlit run frontend/app.py
```

Frontend:

`http://localhost:8501`

## Retrieval Evaluation

The RAG pipeline uses the SQuAD dataset for document retrieval experiments.

Retrieval performance is evaluated using:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)

The validation retrieval benchmark currently reports:

| Metric | Result |
|---|---:|
| Questions Evaluated | 10,570 |
| Top K | 5 |
| Recall@5 | 0.8588 |
| Precision@5 | 0.1905 |
| MRR | 0.7184 |

These results represent an in-corpus validation benchmark: the evaluated questions and their expected evidence paragraphs come from the same validation corpus used to build the retrieval index. They should not be interpreted as unseen-document generalization results.

Retrieval error analysis is available in:

`docs/reports/retrieval_error_analysis.json`

## Testing

The project includes **99 automated tests** covering backend evaluation, data ingestion, preprocessing, embeddings, vector storage, retrieval, retrieval evaluation, and evidence annotation.

Tests must be executed in their respective Python environments.

### 1. Backend and Evidence Annotation Tests

```bash
source venv/bin/activate

python -m pytest -v \
tests/test_accuracy_judge.py \
tests/test_completeness_judge.py \
tests/test_evaluation_orchestrator.py \
tests/test_evidence_annotation.py \
tests/test_hallucination_judge.py \
tests/test_relevance_judge.py \
tests/test_verdict_agent.py
```

### 2. RAG and Retrieval Tests

```bash
source venv-embeddings/bin/activate

python -m pytest -v \
tests/test_dataset_loader.py \
tests/test_preprocessor.py \
tests/test_embedding_model.py \
tests/test_numpy_vector_store.py \
tests/test_indexing_pipeline.py \
tests/test_retrieval_engine.py \
tests/test_retrieval_evaluator.py \
tests/test_retrieval_error_analysis.py
```

The frontend is manually verified through the Streamlit application and FastAPI backend.
