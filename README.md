# PaperPal

PaperPal is a Retrieval-Augmented Generation (RAG) system for helping students read research papers.

## What this MVP includes

- PDF ingestion and chunking
- Dense retrieval with FAISS + sentence-transformers
- Generation with OpenAI API / Claude API / local Ollama
- Streamlit UI for QA
- Citation display from retrieved chunks

## Quick Start

## 1) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Prepare papers

Put your PDF files into:

`data/papers/`

## 3) Build vector index

```bash
python src/ingest.py
```

Expected output:

`Done. Indexed XXX chunks from YYY PDFs.`

## 4) Start app

Create local env file:

```bash
cp .env.example .env
```

Then edit `.env` and fill your `ANTHROPIC_API_KEY`.
If using OpenAI, set:

- `GENERATOR_PROVIDER=openai`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini` (or another available model)

Then start app:

```bash
streamlit run app.py
```

Then open the local URL shown in terminal.

## Optional: local model fallback (Ollama)

If you want local inference:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running on:

`http://localhost:11434`

If using Ollama, set `GENERATOR_PROVIDER=ollama` in `.env`.

## Suggested demo questions

- What is the main problem this paper addresses?
- What is the proposed method?
- What are the key contributions?
- What are limitations mentioned in this paper?

## Minimal evaluation plan (for report)

- Create 5-10 questions
- For each answer, manually rate:
  - relevance (1-5)
  - correctness (1-5)
  - citation support (yes/no)
- Report average scores and 1-2 failure cases

## Project structure

```text
.
├── app.py
├── requirements.txt
├── src
│   ├── config.py
│   ├── ingest.py
│   └── rag.py
├── data
│   └── papers
└── report.md
```

## Tomorrow plan (you said)

1. Push to GitHub
2. Deploy Streamlit service on GCP
3. Prepare one-page presentation slide
