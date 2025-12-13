import time
import json
import random
import numpy as np
from typing import List, Dict

from .agents import SupervisorAgent
from .ingestion import emb_model
from . import config

# -----------------------------
# CONFIG
# -----------------------------
NUM_RANDOM_QS = 10
TOP_K = 5

RANDOM_QUESTIONS = [
    "Does the manual provide step-by-step repair instructions?",
    "What safety precautions are mentioned?",
    "Does the document describe disassembly steps?",
    "Are troubleshooting steps included?",
    "Does the manual mention required tools?",
    "Is warranty information provided?",
    "Does it include diagrams?",
    "Is battery replacement explained?",
    "Are software repair steps included?",
    "Does it mention authorized repair conditions?"
]

# -----------------------------
# INLINE LABELED DATA
# (ground truth for Precision / Recall)
# -----------------------------
LABELED_QA = {
    "Does the manual mention required tools?": {
        "WT7200CV_first_15_pages.pdf"
    },
    "Is warranty information provided?": {
        "WT7200CV_first_15_pages.pdf"
    },
    "Does the document describe disassembly steps?": {
        "WT7200CV_first_15_pages.pdf"
    }
}

# -----------------------------
# METRICS
# -----------------------------
def hallucination_rate(results: List[Dict]) -> float:
    if not results:
        return 0.0

    hallucinated = 0
    for r in results:
        retrieved_sources = {
            c.get("metadata", {}).get("source")
            for c in r.get("retrieval", {}).get("chunks", [])
        }

        evidence = r.get("answer", {}).get("evidence", [])
        used_sources = {
            e.get("source") for e in evidence if isinstance(e, dict)
        }

        if not retrieved_sources.intersection(used_sources):
            hallucinated += 1

    return hallucinated / len(results)


def embedding_diagnostics(query: str, chunks: List[Dict]) -> Dict:
    try:
        q_emb = emb_model.encode(query, normalize_embeddings=True)

        scores = []
        for c in chunks:
            text = c.get("metadata", {}).get("text", "")
            if not text:
                continue

            c_emb = emb_model.encode(text, normalize_embeddings=True)
            sim = float(
                np.dot(q_emb, c_emb)
                / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
            )
            scores.append(sim)

        if not scores:
            return {"mean": None, "max": None, "min": None}

        return {
            "mean": round(float(np.mean(scores)), 3),
            "max": round(float(np.max(scores)), 3),
            "min": round(float(np.min(scores)), 3),
        }

    except Exception:
        return {"mean": None, "max": None, "min": None}


def compute_precision_recall(results: List[Dict]) -> Dict:
    precisions, recalls = [], []

    for r in results:
        query = r.get("plan", {}).get("query")
        if query not in LABELED_QA:
            continue

        relevant = LABELED_QA[query]
        retrieved = {
            c.get("metadata", {}).get("source")
            for c in r.get("retrieval", {}).get("chunks", [])
        }

        if not retrieved:
            continue

        tp = len(retrieved & relevant)

        precision = tp / len(retrieved)
        recall = tp / len(relevant)

        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        return {"precision": None, "recall": None}

    return {
        "precision": round(float(np.mean(precisions)), 3),
        "recall": round(float(np.mean(recalls)), 3),
    }


# -----------------------------
# MAIN EVAL
# -----------------------------
def run_evaluation() -> Dict:
    config.EVALUATION_MODE = True  # 🔒 disable LLM calls

    agent = SupervisorAgent()
    results = []
    latencies = []

    sample_qs = random.sample(
        RANDOM_QUESTIONS, min(NUM_RANDOM_QS, len(RANDOM_QUESTIONS))
    )

    for q in sample_qs:
        start = time.perf_counter()
        out = agent.handle_question(q)
        latencies.append(time.perf_counter() - start)
        results.append(out)

    halluc_rate = hallucination_rate(results)
    pr = compute_precision_recall(results)

    emb_stats = embedding_diagnostics(
        sample_qs[0],
        results[0].get("retrieval", {}).get("chunks", [])[:TOP_K]
        if results else []
    )

    report = {
        "hallucination_rate": round(halluc_rate, 3),
        "latency": {
            "avg": round(float(np.mean(latencies)), 3),
            "min": round(float(np.min(latencies)), 3),
            "max": round(float(np.max(latencies)), 3),
        },
        "embedding_similarity": emb_stats,
        "retrieval_precision": pr["precision"],
        "retrieval_recall": pr["recall"],
        "num_questions": len(sample_qs),
    }

    config.EVALUATION_MODE = False  # 🔓 restore
    return report


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
