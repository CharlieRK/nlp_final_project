import streamlit as st

from src.rag import RagEngine


st.set_page_config(page_title="PaperPal", layout="wide")
st.title("PaperPal: Local RAG for Paper Reading")
st.caption("Ask questions about your paper corpus. Answers are grounded with citations.")

if "engine" not in st.session_state:
    try:
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
