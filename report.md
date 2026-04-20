# PaperPal: Local RAG System for Assisting Research Paper Reading

## 1. Introduction

Reading NLP/ML papers is challenging for students due to dense technical language and high prerequisite knowledge.  
This project proposes PaperPal, a local RAG system that supports question answering over paper content with grounded citations and student-friendly responses.

## 2. Objective

The objectives of this project are:

- Build an end-to-end local RAG pipeline
- Support paper-based QA with evidence citations
- Provide concise and understandable explanations
- Deliver a runnable demo interface

## 3. System Design

The system follows a standard RAG pipeline:

1. **Ingestion**: Parse PDF papers and split text into overlapping chunks
2. **Embedding**: Encode chunks with a sentence-transformer model
3. **Retrieval**: Use FAISS to retrieve top-k relevant chunks
4. **Generation**: Use a local LLM (via Ollama) to generate grounded answers
5. **UI**: Streamlit interface for interactive querying and citation display

## 4. Implementation

### 4.1 Data

- Source: local PDF files in `data/papers/`
- Scope for MVP: small subset of papers (due to time constraints)
- Metadata: paper title, file name, and chunk id

### 4.2 Models and Tools

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS
- LLM: local Ollama model (default `qwen2.5:3b`)
- Frontend: Streamlit

### 4.3 Output format

For each question, the system returns:

- A concise answer
- Retrieved evidence chunks as citations

## 5. Evaluation

Given limited time, we perform a lightweight manual evaluation on 5-10 questions.

Metrics:

- **Relevance (1-5)**: Is retrieved evidence relevant to the question?
- **Correctness (1-5)**: Is the final answer factually aligned with context?
- **Grounding (Yes/No)**: Is the answer supported by shown citations?

| Question ID | Relevance | Correctness | Grounded | Notes |
|---|---:|---:|---|---|
| Q1 |  |  |  |  |
| Q2 |  |  |  |  |
| Q3 |  |  |  |  |
| Q4 |  |  |  |  |
| Q5 |  |  |  |  |

## 6. Results and Discussion

Preliminary observations:

- The system can answer high-level questions about problem, method, and contributions.
- Citation display improves interpretability and trust.
- Errors occur when retrieved chunks are incomplete or not specific enough.

Failure cases and reasons:

- [Fill with 1-2 examples from your tests]

## 7. Limitations

- Small corpus size for MVP
- No reranker or hybrid retrieval in current version
- Limited quantitative automatic metrics

## 8. Future Work

- Scale to larger corpus (1000+ papers)
- Add BM25+dense hybrid retrieval and reranking
- Introduce structured section-aware chunking
- Improve latency and UI interaction quality

## 9. Conclusion

PaperPal demonstrates that a local RAG pipeline can assist paper reading with grounded answers and citations.  
Despite reduced scope for the one-day MVP, the prototype achieves a working end-to-end workflow and provides a strong foundation for further extension.
