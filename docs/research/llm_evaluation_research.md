# Research Notes: LLM Evaluation and Hallucination Detection

## 1. Introduction

Large Language Models can generate fluent and convincing responses that may contain incorrect, unsupported, incomplete, or irrelevant information.

An AI Response Validation System evaluates generated responses across multiple quality dimensions and helps identify potentially unsupported claims.

## 2. Key Evaluation Dimensions

### 2.1 Relevance

Measures whether the generated response addresses the user's question.

### 2.2 Accuracy

Measures whether the response agrees with available reference information.

### 2.3 Completeness

Measures whether the response covers the important concepts required by the question.

### 2.4 Faithfulness and Hallucination Detection

Measures whether claims in the generated response are supported by the supplied source information.

An unsupported claim is not automatically false. It may require additional evidence or human review.

## 3. LLM-as-a-Judge

LLM-as-a-Judge is an evaluation approach in which a language model assesses another model's response against defined criteria.

Possible evaluation criteria include:

- Factual accuracy
- Relevance
- Completeness
- Evidence support
- Clarity

A production implementation should use structured evaluation prompts, consistent scoring rubrics, and human-reviewed test cases.

**Project status:** This project currently implements rule-based lexical evaluation judges. It does not claim to have a fully implemented LLM-as-a-Judge system.

## 4. Retrieval-Augmented Generation (RAG)

RAG retrieves relevant information from an external knowledge source before generating or evaluating an answer.

The project's retrieval pipeline includes:

1. Dataset loading
2. Text preprocessing and chunking
3. Sentence Transformer embeddings
4. NumPy-based vector storage
5. Cosine-similarity retrieval

Retrieved passages can provide evidence for evaluating AI-generated responses.

## 5. Retrieval Evaluation Metrics

### Recall@K

Measures whether the expected relevant passage appears among the top K retrieved results.

### Precision@K

Measures the proportion of the top K retrieved results that are relevant.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant result appears in the ranked retrieval list.

These metrics evaluate retrieval performance, not the factual correctness of the final generated answer.

## 6. Retrieval Error Analysis

Retrieval errors can be categorized into:

- Rank-1 success: The expected passage appears first.
- Lower-rank success: The expected passage appears below rank one but within the retrieved results.
- Retrieval failure: The expected passage is not retrieved.

Exact paragraph matching is useful for benchmarking but may classify a result as a failure even when another retrieved passage contains the correct answer.

## 7. Human Evidence Annotation

Human review can assess whether retrieved passages actually support an expected answer.

The project defines four annotation labels:

- supported
- partially_supported
- unsupported
- insufficient_evidence

These labels help distinguish direct evidence support from partial or missing evidence.

## 8. Benchmarking

The project evaluates retrieval against the SQuAD validation question set.

Current recorded results:

- Questions evaluated: 10,570
- Top K: 5
- Recall@5: 0.8588
- Precision@5: 0.1905
- MRR: 0.7184

**Important limitation:** The validation questions were evaluated against an index built from the same validation corpus containing their expected passages. These results represent an in-corpus retrieval benchmark and should not be interpreted as unseen-document generalization performance.

## 9. Current Limitations

- Lexical overlap is not equivalent to semantic understanding.
- Matching words does not prove factual support.
- Paraphrased answers may be missed by exact-string matching.
- Retrieved evidence may be relevant without matching the exact expected paragraph.
- Retrieval performance depends on corpus quality, chunking, embeddings, and retrieval configuration.
- Human annotation requires a consistent review process.

## 10. Future Improvements

Potential improvements include:

- Semantic similarity-based evaluation
- Natural Language Inference (NLI) for entailment and contradiction detection
- Claim-level evidence verification
- LLM-based judging with structured rubrics
- Human-reviewed evaluation datasets
- Confidence calibration
- Testing on genuinely unseen documents
- Automated evaluation reports and monitoring

## 11. Conclusion

Reliable AI response evaluation requires more than checking whether a response contains matching words. It requires evidence retrieval, claim-level verification, multiple evaluation dimensions, and human review where evidence is uncertain.

This project establishes a modular baseline for these capabilities and provides a foundation for more advanced semantic and model-based evaluation.
