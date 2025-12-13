from .config import EMBEDDING_MODEL, CROSS_ENCODER, TOP_K, CHROMA_DIR
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# lazy Chroma client + collection helpers
_chroma_client = None
def get_chroma_client():
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    try:
        settings = chromadb.config.Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(CHROMA_DIR))
        _chroma_client = chromadb.Client(settings=settings)
        logger.info("Chroma persistent client created at %s", CHROMA_DIR)
    except Exception as e:
        logger.warning("Persistent Chroma client failed (%s). Trying default client. Error: %s", type(e).__name__, e)
        _chroma_client = chromadb.Client()
    return _chroma_client

def get_collection(name="documents"):
    client = get_chroma_client()
    try:
        # modern API: get_or_create_collection
        return client.get_or_create_collection(name=name)
    except TypeError:
        # fallback older signatures
        try:
            return client.get_collection(name)
        except Exception:
            return client.get_or_create_collection(name=name)

# initialize models (lazy but load once)
emb_model = SentenceTransformer(EMBEDDING_MODEL)
try:
    cross_encoder = CrossEncoder(CROSS_ENCODER)
except Exception:
    cross_encoder = None
    logger.warning("Cross-encoder not available; re-ranking disabled unless model installed.")

# replace in retriever.py

def vector_similarity_query(query: str, k: int = TOP_K, min_chars: int = 50):
    """
    Query chroma, return list of candidate chunks as:
      [{"id": <str_or_none>, "score": <0..1 float>, "metadata": {...}}, ...]
    Filters out chunks with very short text (TOC/header noise) using min_chars.
    """
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    coll = get_collection("documents")

    try:
        # modern API - request documents, metadatas, distances
        results = coll.query(query_embeddings=q_emb.tolist(), n_results=k * 3, include=['metadatas','distances','documents'])
    except TypeError:
        # fallback for older signatures
        results = coll.query(query_embeddings=q_emb.tolist(), n_results=k * 3, include=['metadatas','distances'])

    # results format can contain lists per query; we handle single-query case
    metadatas_list = results.get('metadatas', [])
    distances_list = results.get('distances', [])
    docs_list = results.get('documents', [])
    ids_list = results.get('ids', None)  # may not exist

    candidates = []
    # results are per-query; we'll flatten first query (index 0) if present
    mm = metadatas_list[0] if metadatas_list else []
    dd = distances_list[0] if distances_list else []
    docs = docs_list[0] if docs_list else []
    ids = ids_list[0] if ids_list else [None] * len(mm)

    for idx, (meta, dist, doc_text, _id) in enumerate(zip(mm, dd, docs, ids)):
        text = meta.get('text') if isinstance(meta, dict) else doc_text or ""
        if not text:
            text = doc_text or ""
        # filter out tiny chunks/TOC lines
        if len(text.strip()) < min_chars:
            continue
        # convert distance -> similarity (cosine-like)
        sim = None
        if dist is not None:
            # chroma returns L2 distance for some settings; convert to similarity-ish
            sim = 1.0 / (1.0 + float(dist))
        candidates.append({
            "id": _id or meta.get('chunk_id') or meta.get('id'),
            "score": sim,
            "metadata": meta
        })

    # sort by score (None -> -inf)
    candidates = sorted([c for c in candidates if c.get('score') is not None],
                        key=lambda x: x['score'], reverse=True)
    # return top-k
    return candidates[:k]


@lru_cache(maxsize=8)
def build_bm25_for_collection(name="documents"):
    coll = get_collection(name)
    all_meta = coll.get(include=['metadatas','documents'])
    metadatas = all_meta.get('metadatas', [])[0] if all_meta.get('metadatas') else []
    docs = []
    doc_meta_map = []
    for m in metadatas:
        text = None
        if isinstance(m, dict):
            text = m.get('summary') or m.get('text') or ""
        elif isinstance(m, str):
            text = m
        text = text or ""
        docs.append(text.split())
        doc_meta_map.append(m)
    if not docs:
        return None
    bm25 = BM25Okapi(docs)
    return {"bm25": bm25, "meta": doc_meta_map, "raw_docs": docs}

def bm25_query(query:str, k:int=TOP_K):
    coll = get_collection("documents")
    # get all metadatas and documents (API may vary by chroma version)
    try:
        all_meta = coll.get(include=['metadatas','documents'])
    except TypeError:
        # older/newer clients might require different call
        all_meta = coll.get()

    raw_metas = all_meta.get('metadatas', []) if isinstance(all_meta, dict) else []
    raw_docs = all_meta.get('documents', []) if isinstance(all_meta, dict) else []

    # Build list of texts robustly: prefer metadata['text'], then documents entry, then string meta
    docs = []
    for i, m in enumerate(raw_metas):
        if isinstance(m, dict):
            txt = m.get('text') or m.get('summary') or raw_docs[i] if i < len(raw_docs) else ""
        else:
            # m might be a plain string (older Chroma)
            txt = m or (raw_docs[i] if i < len(raw_docs) else "")
        docs.append(txt or "")

    tokenized = [d.split() for d in docs]
    if not tokenized:
        return []

    bm25 = BM25Okapi(tokenized)
    topn_texts = bm25.get_top_n(query.split(), docs, n=k)

    # map topn_texts back to canonical candidate objects with consistent metadata shape
    items=[]
    for txt in topn_texts:
        # find first matching metadata index
        found = False
        for idx, m in enumerate(raw_metas):
            meta_dict = m if isinstance(m, dict) else {"text": m}
            # compare normalized text snippets to match (use startswith to be robust)
            candidate_text = meta_dict.get('text') or (raw_docs[idx] if idx < len(raw_docs) else "")
            if candidate_text and candidate_text.strip().startswith(txt.strip()[:80]):
                items.append({"id": None, "score": None, "metadata": meta_dict})
                found = True
                break
        if not found:
            # fallback: create metadata with bare text
            items.append({"id": None, "score": None, "metadata": {"text": txt}})
    return items



def rerank(query: str, candidates: List[Dict], topk: int = TOP_K):
    if cross_encoder is None or not candidates:
        return candidates[:topk]
    texts = [c['metadata'].get('text','')[:1000] for c in candidates]
    pairs = [[query, t] for t in texts]
    scores = cross_encoder.predict(pairs)
    # normalize to 0..1
    smin, smax = float(scores.min()), float(scores.max())
    rng = smax - smin if smax != smin else 1.0
    for c, s in zip(candidates, scores):
        norm = (float(s) - smin) / rng
        c['rerank_score'] = norm
    return sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)[:topk]

def hybrid_retrieve(query:str, k:int=TOP_K):
    v = vector_similarity_query(query, k=50)
    b = bm25_query(query, k=50)
    text_map = {}
    for item in v + b:
        text = item['metadata']['text']
        if text not in text_map:
            text_map[text] = item
        else:
            if item.get('score',0) and (text_map[text].get('score',0) < item.get('score',0)):
                text_map[text] = item
    merged = list(text_map.values())
    reranked = rerank(query, merged, topk=k)
    diagnostics = {
        "num_vector_candidates": len(v),
        "num_bm25_candidates": len(b),
        "merged_candidates": len(merged)
    }
    return reranked, diagnostics


