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

## Run with Docker + Streamlit (no GCP)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine).

**Option A — Docker Compose (recommended)**

```bash
cd nlp6120_final_project
docker compose up --build
```

Open **http://localhost:8501** (first start runs `ingest.py` inside the container and may take a few minutes).

Stop: `Ctrl+C` or `docker compose down`.

**Option B — Docker only**

```bash
docker build -t paperpal:latest .
docker run --rm -p 8501:8080 -e GENERATOR_PROVIDER=local paperpal:latest
```

Open **http://localhost:8501**.

Notes:

- Default generator in code is **`local`** (no API keys). Override with `-e` if needed.
- Put PDFs in `data/papers/` before `docker build`, or use the compose volume mount and **rebuild** after adding papers (`docker compose up --build`).

## Deploy online with Streamlit Community Cloud (no GCP)

1. Push this repo to **GitHub** (PDFs under `data/papers/` must be committed so the cloud app can read them).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) → **Sign in** → **New app**.
3. Pick your **repository** and **branch** (`main`).
4. **Main file path:** `app.py`
5. **App URL:** Cloud assigns `https://<your-app>.streamlit.app` (you can rename in settings).

On first open, the app runs **`ingest.py` automatically** if the FAISS index is missing (same as Docker), so the **first load can take several minutes** (downloads embedding weights + builds index).

**Optional — Secrets** (only if you switch off `local` mode):  
App settings → **Secrets** → TOML, e.g.:

```toml
GENERATOR_PROVIDER = "local"
```

**Note:** Free tier has **memory/time limits**. If the app crashes during ingest, try fewer PDFs in `data/papers/` or upgrade the Streamlit workspace. `runtime.txt` pins **Python 3.10.12** for compatibility.

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
├── Dockerfile
├── docker-compose.yml
├── runtime.txt
├── requirements.txt
├── src
│   ├── config.py
│   ├── ingest.py
│   └── rag.py
├── data
│   └── papers
└── report.md
```
