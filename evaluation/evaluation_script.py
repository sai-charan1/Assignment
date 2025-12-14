import time
import random
import os
import json
from agents.supervisor_agent import invoke_supervisor
from dotenv import load_dotenv
load_dotenv()

LABELS_PATH = "evaluation/labels.json"


def load_labels():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "eval/labels.json not found.\n"
            "Expected format:\n"
            "[{'question': str, 'answer': str, 'relevant_sources': [str]}]"
        )
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_tests(n_random=10, vectorstore=None, docs=None):
    labels = load_labels()
    if len(labels) == 0:
        raise ValueError("labels.json is empty")

    queries = random.sample(labels, min(n_random, len(labels)))

    total_latency = 0.0
    hallucinations = 0
    retrieval_hits = 0

    detailed_stats = []

    for item in queries:
        question = item["question"]
        gt_sources = set(item.get("relevant_sources", []))

        start = time.perf_counter()
        res = invoke_supervisor(question, vectorstore=vectorstore, docs=docs)
        latency = time.perf_counter() - start

        total_latency += latency

        # ---------- Answer & Evidence ----------
        answer_text = res.get("answer", "")
        evidence = res.get("evidence_used", [])
        top_chunks = res.get("top_chunks", [])

        # ---------- Hallucination ----------
        # hallucination = answer given but no evidence
        is_hallucination = bool(answer_text.strip()) and len(evidence) == 0
        if is_hallucination:
            hallucinations += 1

        # ---------- Retrieval accuracy ----------
        retrieved_sources = {c.get("source") for c in top_chunks if c.get("source")}
        if gt_sources.intersection(retrieved_sources):
            retrieval_hits += 1

        detailed_stats.append({
            "question": question,
            "latency_sec": round(latency, 3),
            "hallucination": is_hallucination,
            "retrieved_sources": list(retrieved_sources),
            "ground_truth_sources": list(gt_sources)
        })

    n = len(queries)

    metrics = {
        "num_queries": n,
        "average_latency_sec": round(total_latency / n, 3),
        "hallucination_rate": round((hallucinations / n) * 100, 2),
        "retrieval_hit_rate": round((retrieval_hits / n) * 100, 2),
    }

    return metrics, detailed_stats
