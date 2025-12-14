import sys
import os
from pathlib import Path

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# ---- Project imports ----
from agents.supervisor_agent import invoke_supervisor
from ingestion.ingestion import extract_and_chunk_docs, create_vectorstore
from evaluation.evaluation_script import run_tests

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------
# Configuration
# --------------------
UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "chromadb/ai_assignment")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="GenAI Analyst", layout="wide")
st.title("🧠 GenAI Analyst")
st.caption("AI Document Analyst using RAG + Multi-Agent Reasoning")

st.markdown("---")

# ====================
# Upload Documents
# ====================
st.header("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        file_path = UPLOAD_DIR / f.name
        with open(file_path, "wb") as fh:
            fh.write(f.getbuffer())
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully")

st.markdown("---")

# ====================
# Ingest Documents
# ====================
st.header("📚 Ingest & Index Documents")

if st.button("Ingest Documents"):
    paths = [
        str(p)
        for p in UPLOAD_DIR.glob("*")
        if p.suffix.lower() in {".pdf", ".txt"}
    ]

    if not paths:
        st.warning("No documents found to ingest")
    else:
        with st.spinner("Processing documents..."):
            docs = extract_and_chunk_docs(paths)
            create_vectorstore(docs)
            st.session_state["docs"] = docs

        st.success(f"Ingested and indexed {len(docs)} chunks")

st.markdown("---")

# ====================
# Ask Question
# ====================
st.header("❓ Ask a Question")

question = st.text_input("Enter your question")

if st.button("Get Answer") and question.strip():

    if not os.path.exists(CHROMA_DIR):
        st.error("Vector store not found. Please ingest documents first.")
    elif "docs" not in st.session_state:
        st.error("Documents not loaded. Please re-ingest.")
    else:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

        with st.spinner("Generating answer..."):
            try:
                result = invoke_supervisor(
                    question=question,
                    vectorstore=vectordb,
                    docs=st.session_state["docs"]
                )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("✅ Final Answer")
                    st.json(result)

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

# ====================
# Evaluation Section
# ====================
st.header("🧪 Evaluation")

if st.button("Run Evaluation"):

    if not os.path.exists(CHROMA_DIR) or "docs" not in st.session_state:
        st.error("Please ingest documents before running evaluation.")
    else:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

        with st.spinner("Running evaluation on sample queries..."):
            metrics, details = run_tests(
                n_random=10,
                vectorstore=vectordb,
                docs=st.session_state["docs"]
            )

        st.success("Evaluation completed")

        # ---- KPI Metrics ----
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Latency (s)", round(metrics["average_latency_sec"], 3))
        col2.metric("Hallucination Rate", f'{metrics["hallucination_rate"]}%')
        col3.metric("Retrieval Hit Rate", f'{metrics["retrieval_hit_rate"]}%')

        st.markdown("### 📊 Overall Metrics")
        st.dataframe(
            pd.DataFrame([metrics]),
            use_container_width=True,
            hide_index=True
        )

        with st.expander("📋 Per-Query Evaluation Details", expanded=True):
            st.dataframe(
                pd.DataFrame(details),
                use_container_width=True
            )

st.markdown("---")
st.caption("GenAI Analyst | RAG + Hybrid Retrieval + Evaluation Framework")
