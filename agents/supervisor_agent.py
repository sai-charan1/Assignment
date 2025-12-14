# agents/supervisor_agent.py  (replace only the invoke_supervisor function)
import json
import re
from typing import Any, Dict

from agents.query_analyzer_agent import query_analyzer_agent
from agents.answer_agent import answer_agent
from ingestion.retrieval import HybridRetriever

# helper: extract first JSON object from a model string (fallback)
def _extract_first_json(s: str):
    if not isinstance(s, str):
        return None
    # quick attempt: direct load
    try:
        return json.loads(s)
    except Exception:
        pass
    # fallback: find {...} block (greedy but practical)
    m = re.search(r"\{(?:[^{}]|(?R))*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            # try to fix common single-quotes -> double-quotes
            txt = m.group(0).replace("'", "\"")
            try:
                return json.loads(txt)
            except Exception:
                return None
    return None

def invoke_supervisor(question: str, vectorstore, docs, max_context_chars: int = 28000) -> Dict[str, Any]:
    """
    Orchestrate Query Analyzer -> Retrieval -> Answer Agent.
    Returns structured result where 'answer' appears first, then 'top_chunks' and retrieval diagnostics.
    """

    # 1) Query analysis (produce plan JSON string)
    plan_state = query_analyzer_agent.invoke({"messages": [{"role": "user", "content": question}]})
    try:
        plan_msg = plan_state["messages"][-1].content
        plan = json.loads(plan_msg)
    except Exception:
        # if plan parsing fails, fall back to safe defaults
        plan = {"retrieval_strategy": "hybrid", "top_k": 5, "query": question}

    retrieval_strategy = plan.get("retrieval_strategy", "hybrid")
    top_k = int(plan.get("top_k", 5))
    query = question


    # 2) Retrieval (hybrid retriever expects original docs list)
    retriever = HybridRetriever(vectorstore, docs)
    top_chunks, diagnostics = retriever.retrieve(query, top_k=top_k)

    # 3) Build combined context (annotate with source) and guard length
    pieces = []
    for c in top_chunks:
        src = c.get("source", "unknown")
        text = c.get("text", "") or ""
        # keep short excerpts per chunk to avoid token explosion
        pieces.append(f"SOURCE: {src}\n{text.strip()}")
    combined_context = "\n\n---\n\n".join(pieces)

    # Truncate context if too long: keep head + tail to preserve useful bits
    if len(combined_context) > max_context_chars:
        head = combined_context[: max_context_chars // 2]
        tail = combined_context[- (max_context_chars // 2) :]
        combined_context = head + "\n\n...<<TRUNCATED>>...\n\n" + tail

    # 4) Prepare a plain-string instruction prompt (answer_agent expects a normal message string)
    prompt = (
        "You are an expert analyst. Use ONLY the CONTEXT below to answer the QUESTION. "
        "Do NOT invent facts. If the context does not contain the information, explicitly say what is missing. "
        "Cite sources from the CONTEXT (use the 'SOURCE: filename' markers).\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT (top {len(top_chunks)} chunks):\n{combined_context}\n\n"
        "OUTPUT: Return STRICT JSON only (no surrounding prose) with the exact fields:\n"
        '{"answer":"", "evidence_used":[{"source":"<filename>","excerpt":"<short excerpt>"}], '
        '"missing_information":"", "confidence_score":0.0}\n\n'
        "Evidence items should reference chunks in the CONTEXT; confidence must be a number between 0 and 1.\n"
        "If multiple relevant excerpts exist, include up to 5 evidence entries.\n"
        "Return only valid JSON."
    )

    # 5) Call answer agent with a plain string message (not a dict) to avoid pydantic/list validation errors
    try:
        answer_state = answer_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        raw_answer = answer_state["messages"][-1].content
    except Exception as e:
        return {"error": f"Answer agent invocation failed: {e}", "top_chunks": top_chunks, "retrieval_diagnostics": diagnostics}

    # 6) Parse JSON answer (robust)
    parsed = _extract_first_json(raw_answer)
    if not parsed:
        # last resort: return raw text under answer and include diagnostics
        return {
            "answer": raw_answer if isinstance(raw_answer, str) else str(raw_answer),
            "evidence_used": [],
            "missing_information": "Could not parse structured JSON from the model.",
            "confidence_score": 0.0,
            "top_chunks": top_chunks,
            "retrieval_diagnostics": diagnostics,
        }

    # 7) Normalize response fields
    answer_text = parsed.get("answer", "")
    evidence = parsed.get("evidence_used", []) or []
    missing = parsed.get("missing_information", parsed.get("missing_info", ""))
    confidence = parsed.get("confidence_score", parsed.get("confidence", 0.0))
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    # 8) Final structured return (answer first)
    return {
        "answer": answer_text,
        "evidence_used": evidence,
        "missing_information": missing,
        "confidence_score": confidence,
        "top_chunks": top_chunks,
        "retrieval_diagnostics": diagnostics,
        "plan": plan,
    }
