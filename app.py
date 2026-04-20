import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.config import CHUNKS_FILE, VECTOR_DIR
from src.rag import RagEngine


def _ensure_vector_index() -> None:
    """Build FAISS index on first run (e.g. Streamlit Community Cloud has no pre-built index)."""
    index_path = VECTOR_DIR / "index.faiss"
    if index_path.exists() and CHUNKS_FILE.exists():
        return
    root = Path(__file__).resolve().parent
    ingest_script = root / "src" / "ingest.py"
    result = subprocess.run(
        [sys.executable, str(ingest_script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr or result.stdout or f"ingest failed with code {result.returncode}"
        )


st.set_page_config(page_title="PaperPal", layout="wide")
st.title("PaperPal: Local RAG for Paper Reading")
st.caption("Ask questions about your paper corpus. Answers are grounded with citations.")

if "engine" not in st.session_state:
    try:
        with st.spinner("Checking index… (first deploy may build from PDFs, several minutes)"):
            _ensure_vector_index()
        st.session_state.engine = RagEngine()
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.info("Run `python src/ingest.py` after placing PDFs in `data/papers/`.")
        st.stop()

question = st.text_input("Enter your question", placeholder="What is the main contribution of this paper?")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            answer, refs = st.session_state.engine.ask(question.strip())
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.markdown(answer)

    st.subheader("Citations")
    for i, c in enumerate(refs, start=1):
        with st.expander(f"[{i}] {c['paper_title']} - chunk {c['local_chunk_id']}"):
            st.write(c["text"])
