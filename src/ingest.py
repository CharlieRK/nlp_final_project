import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from src.config import (
        CHUNKS_FILE,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_CHUNK_SIZE,
        EMBED_MODEL_NAME,
        PAPERS_DIR,
        VECTOR_DIR,
    )
except ModuleNotFoundError:
    from config import (  # type: ignore
        CHUNKS_FILE,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_CHUNK_SIZE,
        EMBED_MODEL_NAME,
        PAPERS_DIR,
        VECTOR_DIR,
    )


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks() -> List[Dict]:
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    all_chunks = []
    chunk_id = 0

    for pdf_path in tqdm(pdf_files, desc="Reading PDFs"):
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_title": pdf_path.stem,
                    "source_file": pdf_path.name,
                    "local_chunk_id": idx,
                    "text": chunk,
                }
            )
            chunk_id += 1
    return all_chunks


def save_chunks_jsonl(chunks: List[Dict]) -> None:
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def build_vector_index(chunks: List[Dict]) -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))
    np.save(VECTOR_DIR / "embeddings.npy", embeddings)


def main() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_count = len(list(PAPERS_DIR.glob("*.pdf")))
    if pdf_count == 0:
        raise SystemExit(
            "No PDF found in data/papers. Put some papers there first, then rerun ingest."
        )

    chunks = build_chunks()
    if not chunks:
        raise SystemExit("No text extracted from PDFs. Please check your PDF files.")

    save_chunks_jsonl(chunks)
    build_vector_index(chunks)
    print(f"Done. Indexed {len(chunks)} chunks from {pdf_count} PDFs.")


if __name__ == "__main__":
    main()
