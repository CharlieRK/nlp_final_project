# PaperPal: A Retrieval-Augmented Generation System for Research Paper Reading

**Course:** NLP 6120 — Final Project  
**Repository:** [https://github.com/CharlieRK/nlp_final_project](https://github.com/CharlieRK/nlp_final_project)

---

## Abstract

Reading academic papers in NLP and machine learning is difficult for students because of dense terminology and long documents. This project implements **PaperPal**, a retrieval-augmented generation (RAG) application that answers questions over a **locally stored paper corpus** using dense retrieval and citation-backed evidence. Following the course direction toward **~10,000 papers**, the pipeline is built to batch-ingest PDFs from disk, embed chunks, and index them with FAISS—**without** sending raw PDFs to external LLM APIs. For grading and reproducibility, the public GitHub repo includes a **small validation subset** and an example index size; the same ingestion path applies to the full **local ~10k** collection. The app provides a Streamlit interface and optional API-based generators, with a **local API-free extractive fallback** as default for stable demos. The service is containerized with Docker and deployed on **Google Cloud Run**. We report qualitative evaluation and discuss scaling (e.g., index structure at 10k scale) and hybrid retrieval as future work.

---

## 1. Introduction

Students often struggle to locate main contributions, methods, and evaluation details in papers. Generic chatbots may hallucinate or omit sources. RAG addresses this by conditioning answers on retrieved passages and surfacing those passages as citations. PaperPal follows this design: users ask questions; the system retrieves relevant chunks from indexed papers and returns an answer together with expandable citations.

---

## 2. Objectives

1. Implement an end-to-end RAG pipeline: ingestion, embedding, vector search, generation, and UI.  
2. Support **grounded** responses with visible citation chunks.  
3. Allow **paper-targeted** queries (e.g., arXiv-style IDs in the question) to reduce cross-paper confusion.  
4. Deliver a **reproducible** repository, a **live demo** on GCP, and this technical report.

---

## 3. System Architecture

### 3.1 Pipeline

1. **Ingestion:** PDFs under `data/papers/` are read with `pypdf`, normalized, and split into overlapping character chunks (configurable size and overlap).  
2. **Embedding:** Chunks are encoded with `sentence-transformers/all-MiniLM-L6-v2`.  
3. **Indexing:** Embeddings are stored in a **FAISS** `IndexFlatL2` index; chunk metadata (paper title, source filename, local chunk id) is stored alongside for display.  
4. **Retrieval:** The user question is embedded; top-k neighbors are retrieved. Optional **hint extraction** (arXiv-like ids, quoted phrases) restricts or reranks candidates toward the intended paper.  
5. **Generation:** Configurable backends:
   - **`local` (default):** Extractive-style answer from retrieved text (no external API; stable for demos).  
   - **Anthropic / OpenAI / Ollama:** Optional, via environment variables.  
6. **UI:** **Streamlit** app (`app.py`) shows the answer and expandable citation blocks.

### 3.2 Deployment

- **Docker** image built from the included `Dockerfile` (Python 3.10, dependencies from `requirements.txt`, Streamlit on `$PORT`).  
- **Google Cloud Run** runs the container; startup can run `python src/ingest.py` before `streamlit run` so the vector index exists in the container filesystem for the demo.

---

## 4. Data and Experimental Setup

| Item | Value |
|------|--------|
| **Design corpus (local)** | **~10,000 papers** (course target; PDFs kept on local / server disk, ingested offline) |
| **Repository demo subset** | Representative PDFs bundled for reproduction; example run ~**1,340** chunks from a 10-paper batch |
| Embedding model | `all-MiniLM-L6-v2` |
| Retrieval | FAISS, top-k = 4 (default) |

**Framing:** The system is **designed for a local ~10k-paper library** (batch `ingest.py`, chunked embeddings, on-disk index). Instructors and graders can run ingestion on their own full corpus path; the submitted repo demonstrates the **end-to-end path** on a subset so that clone → ingest → query remains lightweight.

---

## 5. Evaluation

We used **manual spot checks** rather than large-scale automatic metrics (time-limited). For each test question we judged:

- **Relevance (1–5):** Do retrieved chunks relate to the question?  
- **Correctness (1–5):** Does the answer align with the cited text?  
- **Grounded (Y/N):** Is the answer supported by shown citations?

Example results from demo sessions (paper-specific queries with arXiv-style ids):

| ID | Question focus | Rel. | Corr. | Grounded | Notes |
|----|----------------|-----:|------:|:--------:|--------|
| Q1 | Method in a specified paper | 4 | 4 | Y | Citations from correct PDF filename / id |
| Q2 | Contribution / takeaway | 4 | 3 | Y | Extractive answer depends on PDF text quality |
| Q3 | Cross-paper comparison | 3 | 3 | Y | Harder; wording matters |

**Observations:**  
- Paper-targeted queries (including document ids in the prompt) improve stability.  
- Some PDFs yield noisier extracted text; answers then reflect that noise.  
- The local extractive mode is **faithful to retrieved spans** but not always as fluent as a large generative model.

---

## 6. Limitations

1. **Scale-up:** At **~10k papers** and millions of chunks, a flat FAISS index and full re-embed may need **sharding, incremental updates, or approximate ANN** (e.g., IVF) for latency and memory—beyond the current flat index demo.  
2. **Chunking:** Uniform character chunks, not full section-aware structure (intro/method/experiments).  
3. **Retrieval:** Dense-only; no BM25 hybrid or cross-encoder reranking in the shipped MVP.  
4. **Generation:** Default local mode prioritizes **availability and traceability** over abstract paraphrasing quality.  
5. **Cost and latency:** Cloud Run cold starts and model loading can make the first request slow; a full local ~10k ingest is **CPU/disk intensive** and is best run on a workstation or batch job, not only in a small Cloud Run container.

---

## 7. Future Work

- Production **batch-ingest** for the full **local ~10k** corpus with incremental indexing and monitoring.  
- **Hybrid retrieval** (BM25 + dense) and **reranking**.  
- **Section-aware** chunking and cleaner PDF text (layout-aware parsers).  
- Stronger **local LLM** (e.g., via Ollama) or API models when keys/budget allow.  
- Systematic **automatic metrics** (e.g., retrieval precision@k, answer faithfulness checks).

---

## 8. Deliverables

| Deliverable | Location / Notes |
|-------------|------------------|
| Source code | GitHub: `CharlieRK/nlp_final_project` |
| README | Setup, ingest, Streamlit, Docker, env vars |
| Live demo | Cloud Run URL *(update after each redeploy)* |
| Container | `Dockerfile`, `.dockerignore` |
| This report | `report.md` (this file) |

---

## 9. Conclusion

PaperPal implements a working RAG stack aligned with a **local ~10,000-paper** design target: offline ingestion, dense retrieval, citation-grounded answers, and a **deployable** Streamlit demo on GCP. The repository proves the full pipeline; scaling the index and hardware for the complete local corpus is the natural next engineering step.

---

## References (illustrative)

Implementation follows standard RAG practice; embedding and vector search use publicly available libraries (Sentence-Transformers, FAISS). Papers in `data/papers/` are included for reproducibility of the demo corpus; cite original works when describing content.
