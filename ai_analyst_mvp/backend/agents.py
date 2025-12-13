# backend/agents.py
from enum import Enum
from .llm_tools import call_llm
from typing import List, Dict, Any, Optional
from .retriever import hybrid_retrieve
from .prompts import load_prompt
import time
import logging
import textwrap

logger = logging.getLogger(__name__)

# Optional LLM caller: if you implement an LLM wrapper (e.g. openai, or local model),
# place a function call_llm(prompt: str, max_tokens: int) -> str in backend/llm_tools.py
# This module will be imported if present; otherwise fallback path is used.
try:
    from .llm_tools import call_llm  # type: ignore
    _has_llm = True
    logger.info("LLM tools detected: will use call_llm for answer generation.")
except Exception:
    _has_llm = False
    logger.info("No LLM tools found: using fallback template generator.")

class QType(Enum):
    FACTUAL = 'factual'
    REASONING = 'reasoning'
    COMPARISON = 'comparison'
    MULTI_HOP = 'multi-hop'
    MISSING = 'missing_data'

class QueryAnalyzerAgent:
    """
    Produces a retrieval plan from the raw user query.
    Plan example:
      {
        "qtype": "factual",
        "num_chunks": 5,
        "query": "<possibly rewritten>",
        "retrieval_strategy": "hybrid",
        "timestamp": 123456.0
      }
    """
    def __init__(self):
        # load any query / rewrite prompts if needed in future
        pass

    def analyze(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        qlower = q.lower()

        # Heuristic classification
        if any(tok in qlower for tok in ['compare', 'versus', ' vs ', 'difference', 'which is better']):
            qtype = QType.COMPARISON
            num_chunks = 8
        elif any(tok in qlower for tok in ['why', 'how', 'explain', 'cause', 'reason']):
            qtype = QType.REASONING
            num_chunks = 12
        elif any(tok in qlower for tok in ['multi-step', 'multi hop', 'multi-hop', 'chain']):
            qtype = QType.MULTI_HOP
            num_chunks = 15
        elif any(tok in qlower for tok in ['missing', 'not contained', 'unknown']):
            qtype = QType.MISSING
            num_chunks = 10
        else:
            # Default/factual
            qtype = QType.FACTUAL
            # short queries -> fewer chunks; long queries -> more chunks
            num_chunks = 5 if len(q.split()) < 8 else 8

        # Simple query rewriting placeholder (could be expanded)
        rewritten = q
        # Safety/fallbacks
        if not q:
            rewritten = ""
            logger.warning("Empty query passed to analyzer; returning safe default plan.")

        plan = {
            "qtype": qtype.value,
            "num_chunks": num_chunks,
            "query": rewritten,
            "retrieval_strategy": "hybrid",
            "timestamp": time.time()
        }
        return plan

class RetrievalAgent:
    """
    Executes retrieval plans using the retriever.hybrid_retrieve function.
    Returns: {"chunks": [...], "diagnostics": {...}, "time": float}
    """
    def __init__(self):
        pass

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        q = plan.get("query", "")
        k = int(plan.get("num_chunks", 5))
        strategy = plan.get("retrieval_strategy", "hybrid")
        start = time.time()

        # Only hybrid implemented in the retriever module; keep strategy value for compatibility
        try:
            chunks, diagnostics = hybrid_retrieve(q, k=k)
            elapsed = time.time() - start
            return {"chunks": chunks, "diagnostics": diagnostics, "time": elapsed, "strategy": strategy}
        except Exception as e:
            logger.exception("RetrievalAgent.execute failed: %s", e)
            elapsed = time.time() - start
            return {
                "chunks": [],
                "diagnostics": {"error": str(e)},
                "time": elapsed,
                "strategy": strategy
            }

class AnswerAgent:
    """
    Produces final structured answer:
      {
        "answer": "<string>",
        "evidence": [ {source, chunk_index, text}, ... ],
        "missing": [ "<bullet>" ],
        "confidence": float
      }
    It will try to use an LLM (if backend/llm_tools.call_llm exists). Otherwise it uses a safe templated generator.
    """
    def __init__(self, prompt_name: str = "answer_prompt.txt"):
        # load the answer-generation prompt template (for when LLM is available)
        try:
            self.prompt_template = load_prompt(prompt_name)
        except Exception as e:
            logger.warning("Could not load answer prompt '%s': %s", prompt_name, e)
            self.prompt_template = None

    def _assemble_evidence(self, chunks: List[Dict], max_chars: int = 800) -> List[Dict]:
        ev = []
        for c in chunks:
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            text = (meta.get("text") or "")[:max_chars]
            ev.append({
                "source": meta.get("source") or meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index"),
                "text": text
            })
        return ev

    def _confidence_from_scores(self, chunks: List[Dict]) -> float:
        # Simple heuristic: use average 'score' if present, otherwise base on number of chunks
        if not chunks:
            return 0.0
        scores = [c.get("score") for c in chunks if c.get("score") is not None]
        if scores:
            avg = sum(scores) / len(scores)
            # normalize (scores assumed in 0..1); clamp
            conf = max(0.0, min(1.0, float(avg)))
            # nudge by number of chunks (more chunks slightly increases confidence)
            conf = conf * 0.85 + min(0.15, 0.03 * len(chunks))
            return round(conf, 3)
        # fallback: small confidence growth with chunk count
        conf = min(0.85, 0.25 + 0.05 * len(chunks))
        return round(conf, 3)

    def _fallback_generate(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Non-LLM fallback: extract top sentences from top chunk(s) and assemble evidence.
        """
        evidence = self._assemble_evidence(chunks, max_chars=1000)
        if not evidence:
            return {
                "answer": "I couldn't find relevant information in the indexed documents.",
                "evidence": [],
                "missing": ["No matching chunks found; try broader query or upload more documents."],
                "confidence": 0.0
            }

        # Pick top chunk (by score if provided, else first)
        top_chunk = None
        if any(c.get("score") is not None for c in chunks):
            # sort preserving original order for tie-breaker
            top_chunk = sorted(chunks, key=lambda x: (-(x.get("score") or 0), x.get("metadata", {}).get("chunk_index", 0)))[0]
        else:
            top_chunk = chunks[0]

        top_text = (top_chunk.get("metadata", {}) .get("text") or "")[:1200]
        # crude sentence extraction
        first_sent = textwrap.shorten(top_text.split('\n')[0], width=400, placeholder='...')
        answer_text = f"(Analyst) {first_sent}"

        # Basic missing info heuristic
        missing = []
        if len(chunks) < 3:
            missing.append("Only a small number of supporting chunks were found; answer may be incomplete.")

        confidence = self._confidence_from_scores(chunks)
        return {
            "answer": answer_text,
            "evidence": evidence[:5],
            "missing": missing,
            "confidence": confidence
        }

    def answer(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Main entrypoint. Attempts to call LLM if available; otherwise falls back to the template generator.
        The produced structure matches the assignment requirements (Answer, Evidence Used, Missing Information, Confidence).
        """
        try:
            # If LLM wrapper exists, build prompt and call it
            if _has_llm and self.prompt_template:
                # Build compact context: top-N chunk texts with short metadata
                top_texts = []
                for c in chunks[:10]:
                    meta = c.get("metadata", {})
                    txt = (meta.get("text") or "").strip()
                    src = meta.get("source") or meta.get("doc_id")
                    idx = meta.get("chunk_index")
                    top_texts.append(f"[{src}#{idx}] {txt[:1000]}")

                context_block = "\n\n---\n\n".join(top_texts)
                # Fill prompt template (prompt should expect placeholders like {query} and {context})
                if "{query}" in self.prompt_template and "{context}" in self.prompt_template:
                    prompt = self.prompt_template.format(query=query, context=context_block)
                else:
                    # naive concatenation if template does not use placeholders
                    prompt = f"{self.prompt_template}\n\nQUESTION:\n{query}\n\nCONTEXT:\n{context_block}"

                # call the LLM; assume call_llm returns a string containing JSON-like or plain text
                try:
                    llm_resp = call_llm(prompt, max_tokens=800)
                    # Best-effort: if LLM returns JSON, try to parse; otherwise include as answer
                    import json
                    try:
                        parsed = json.loads(llm_resp)
                        mapped = {
            "answer": parsed.get("Answer") or parsed.get("answer"),
            "evidence": parsed.get("Evidence Used") or parsed.get("evidence") or [],
            "missing": parsed.get("Missing Information") or parsed.get("missing") or [],
            "confidence": parsed.get("Confidence") or parsed.get("confidence") or 0.0
        }
                        
                        
                        
                        # expect parsed to have keys: answer, evidence, missing, confidence
                        return parsed
                    except Exception:
                        # not JSON — return LLM raw text as answer, and include assembled evidence
                        evidence = self._assemble_evidence(chunks, max_chars=1000)
                        confidence = self._confidence_from_scores(chunks)
                        return {"answer": llm_resp, "evidence": evidence[:5], "missing": [], "confidence": confidence}
                except Exception as e:
                    logger.exception("LLM call failed: %s", e)
                    # fall back to template generator below
                    return self._fallback_generate(query, chunks)

            # No LLM available -> fallback generator
            return self._fallback_generate(query, chunks)

        except Exception as e:
            logger.exception("AnswerAgent.answer failed: %s", e)
            # Safe error response for UI
            return {
                "answer": "An error occurred while producing an answer.",
                "evidence": [],
                "missing": [str(e)],
                "confidence": 0.0
            }


# Optional: Supervisor to orchestrate the three agents
class SupervisorAgent:
    def __init__(self):
        self.analyzer = QueryAnalyzerAgent()
        self.retriever = RetrievalAgent()
        self.answer_agent = AnswerAgent()

    def handle_question(self, question: str) -> Dict[str, Any]:
        plan = self.analyzer.analyze(question)
        retrieval = self.retriever.execute(plan)
        chunks = retrieval.get("chunks", [])
        ans = self.answer_agent.answer(plan.get("query", question), chunks)
        return {"plan": plan, "retrieval": retrieval, "answer": ans}

