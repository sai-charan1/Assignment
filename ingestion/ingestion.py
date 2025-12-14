# ingestion/ingestion.py
import os
import re
import unicodedata
from dotenv import load_dotenv
load_dotenv()

# loader fallbacks
try:
    # preferred if available
    from langchain.document_loaders import UnstructuredPDFLoader as _UnstructuredPDFLoader
    UnstructuredPDFLoader = _UnstructuredPDFLoader
except Exception:
    UnstructuredPDFLoader = None

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

try:
    from langchain_community.document_loaders import TextLoader
except Exception:
    # last-resort fallback
    from langchain.document_loaders import TextLoader

# text splitter (many environments expose this at langchain_text_splitters)
try:
    from langchain_text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

# vectorstore / embeddings (keep your existing Chroma/HF path with fallback)
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    from langchain_chroma import Chroma  # fallback if installed

# Prefer the langchain-huggingface package if available, else fall back
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Global embeddings instance (reuse across calls)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Utility: clean text to remove control characters / weird glyphs
import typing
def clean_text(s: typing.Optional[str]) -> str:
    if not s:
        return ""
    # Normalize unicode
    s = unicodedata.normalize("NFKC", s)
    # Replace non-printable / control characters with space (keep basic unicode ranges)
    s = re.sub(r"[^\t\n\r\u0020-\u007E\u00A0-\uFFFD]+", " ", s)
    # Collapse repeated whitespace to single space (preserve newlines if you want)
    s = re.sub(r"[ \t\v\f\r]+", " ", s)
    s = re.sub(r"\n{2,}", "\n\n", s)  # keep paragraph breaks
    s = s.strip()
    return s

def _load_file(path: str):
    """Robust loader: prefer UnstructuredPDFLoader (if installed), else PyPDFLoader."""
    if path.lower().endswith(".pdf"):
        if UnstructuredPDFLoader is not None:
            loader = UnstructuredPDFLoader(path)
        elif PyPDFLoader is not None:
            loader = PyPDFLoader(path)
        else:
            raise RuntimeError("No PDF loader available in the environment (install unstructured or langchain-community).")
    else:
        try:
            loader = TextLoader(path, encoding="utf-8")
        except Exception:
            # fallback to basic read
            class SimpleTextLoader:
                def __init__(self, path):
                    self.path = path
                def load(self):
                    from langchain import Document
                    with open(self.path, "r", encoding="utf-8", errors="ignore") as fh:
                        txt = fh.read()
                    return [Document(page_content=txt, metadata={"source": os.path.basename(self.path)})]
            loader = SimpleTextLoader(path)
    return loader

def extract_and_chunk_docs(doc_paths, chunk_size=1000, chunk_overlap=200):
    """
    Loads, cleans, and semantically chunks documents.
    Returns list of Document objects (with cleaned page_content and metadata.source)
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for path in doc_paths:
        loader = _load_file(path)
        docs = loader.load()

        # Clean text on each Document before splitting
        for d in docs:
            # ensure page_content exists
            raw = getattr(d, "page_content", None)
            d.page_content = clean_text(raw)
            # ensure metadata has source
            md = getattr(d, "metadata", {}) or {}
            md.setdefault("source", os.path.basename(path))
            d.metadata = md

        # split cleaned docs
        chunks = splitter.split_documents(docs)
        # ensure each chunk has source metadata
        for c in chunks:
            c.metadata.setdefault("source", os.path.basename(path))
        all_chunks.extend(chunks)

    return all_chunks

def create_vectorstore(doc_chunks, persist_directory=None):
    """
    Creates/updates a Chroma vectorstore using the global EMBEDDINGS.
    Deletes and recreates if persist_directory doesn't exist? We persist by default.
    """
    persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chromadb/ai_assignment")
    os.makedirs(persist_directory, exist_ok=True)

    # Chroma.from_documents expects embedding argument name depending on package version;
    # we pass as `embedding` which works for langchain_community's Chroma wrapper.
    vectordb = Chroma.from_documents(
        documents=doc_chunks,
        embedding=EMBEDDINGS,
        persist_directory=persist_directory
    )

    # newer Chroma versions persist automatically; calling persist() is harmless if available
    try:
        vectordb.persist()
    except Exception:
        pass

    return vectordb
