# ingestion.py
from pathlib import Path
from .utils import extract_text_from_pdf, chunk_text_semantic, clean_text, ensure_dir
from .config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import uuid
import json
import logging
import re

logger = logging.getLogger(__name__)

# ensure directories exist
ensure_dir(DATA_DIR)
ensure_dir(CHROMA_DIR)

# Lazy-initialize the Chroma client so server can start even if Chroma settings change
_chroma_client = None

def get_chroma_client():
    """
    Return a Chroma client. Try to create a persistent client first using the modern
    'settings=' keyword. If that fails (e.g. older/newer chroma version mismatch),
    fall back to a default client (in-memory).
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    try:
        # Newer Chroma API expects the settings keyword. This will persist DB to CHROMA_DIR.
        settings = chromadb.config.Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(CHROMA_DIR))
        _chroma_client = chromadb.Client(settings=settings)
        logger.info("Created persistent Chroma client with duckdb+parquet at %s", CHROMA_DIR)
    except Exception as e:
        logger.warning("Failed to create persistent Chroma client (%s). Falling back to default client. Error: %s", type(e).__name__, e)
        try:
            # Fallback: create a default client (may be in-memory depending on chroma version)
            _chroma_client = chromadb.Client()
            logger.info("Created fallback/default Chroma client (in-memory or default local).")
        except Exception as e2:
            # Last resort: raise error
            logger.exception("Unable to create any Chroma client: %s", e2)
            raise

    return _chroma_client

# initialize embedding model once (sentence-transformers)
emb_model = SentenceTransformer(EMBEDDING_MODEL)

# --- chunk cleaning & summary utilities ---
_TOCDATA_RE = re.compile(r"(^[\s\.\-]{3,}$)|(\.{4,})|(^page\s*\d+\b)|(^\d+\s*/\s*\d+\b)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def clean_chunk_text(text: str) -> str:
    """
    Remove TOC-like lines, repeated dots, tiny header/footer lines and normalize whitespace.
    """
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        ln2 = ln.strip()
        if not ln2:
            continue
        if _TOCDATA_RE.search(ln2):
            continue
        # drop lines that are mostly punctuation or extremely short non-informative
        if len(ln2) < 8 and sum(1 for c in ln2 if c.isalnum()) < 2:
            continue
        lines.append(ln2)
    cleaned = " ".join(lines)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned

def chunk_summary(text: str, max_sentences: int = 2) -> str:
    """
    Produce a short 1-2 sentence summary for the chunk (used for BM25 / diagnostics).
    Falls back to a short truncated string when sentences can't be detected.
    """
    if not text:
        return ""
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if len(s.strip()) > 20]
    if not sents:
        return text[:200].strip()
    return " ".join(sents[:max_sentences])

def text_quality(text: str) -> float:
    """
    Very simple quality heuristic: proportion of alphanumeric characters in the chunk.
    """
    if not text:
        return 0.0
    chars = len(text)
    alnum = sum(1 for c in text if c.isalnum())
    return float(alnum) / max(chars, 1)

# --- ingestion function ---
def ingest_file(file_path: Path, doc_id: str = None, metadata: dict = None):
    """
    Extract text, semantically chunk it, clean chunks, compute embeddings and add to Chroma.
    Returns number of chunks created (after filtering).
    """
    # Extract text from file (pdf/text)
    text = extract_text_from_pdf(file_path)
    if not text:
        logger.warning("No text extracted from %s", file_path)
        return 0

    # Semantic chunking (returns list of chunk strings)
    raw_chunks = chunk_text_semantic(text)
    if not raw_chunks:
        logger.warning("chunk_text_semantic returned empty for %s", file_path)
        return 0

    # Process chunks: clean, summarize, compute quality; skip tiny/noisy chunks
    processed_texts = []
    metadatas = []
    ids = []
    for i, raw in enumerate(raw_chunks):
        cleaned = clean_chunk_text(raw)
        if not cleaned or len(cleaned) < 50:
            # skip tiny / TOC-like chunks
            continue
        _id = f"{doc_id or file_path.stem}-{i}-{uuid.uuid4().hex[:6]}"
        meta = {
            "source": str(file_path.name),
            "doc_id": doc_id or file_path.stem,
            "chunk_index": i,
            "chunk_id": _id,
            "text": cleaned,
            "summary": chunk_summary(cleaned, max_sentences=2),
            "quality": round(text_quality(cleaned), 3)
        }
        ids.append(_id)
        metadatas.append(meta)
        processed_texts.append(cleaned)

    if not processed_texts:
        logger.warning("All chunks filtered out for %s", file_path)
        return 0

    # compute embeddings for the cleaned/filtered chunks (batch)
    embeddings = emb_model.encode(processed_texts, show_progress_bar=True, convert_to_numpy=True)

    client = get_chroma_client()

    # obtain/create collection
    try:
        collection = client.get_or_create_collection(name="documents")
    except TypeError:
        collection = client.get_or_create_collection("documents")

    # add to collection (handle API variations)
    try:
        # modern API expects ids, embeddings, metadatas (documents optional)
        collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
    except TypeError:
        try:
            # older API variant
            collection.add(ids=ids, documents=processed_texts, metadatas=metadatas, embeddings=embeddings.tolist())
        except Exception as e:
            logger.exception("Failed to add to Chroma collection: %s", e)
            raise

    # persist if supported
    try:
        client.persist()
    except Exception:
        pass

    logger.info("Ingested %d chunks from %s", len(processed_texts), file_path.name)
    return len(processed_texts)

