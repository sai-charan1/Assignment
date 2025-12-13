import React from "react";

export default function Evaluation({ data }) {
  if (!data) return null;

  const latency = data.latency || {};
  const embedding = data.embedding_similarity || {};

  return (
    <div className="evaluation-card">
      <h2>📊 System Evaluation</h2>

      {/* Hallucination Rate */}
      <div className="metric">
        <span>Hallucination Rate</span>
        <strong>
          {data.hallucination_rate !== null &&
          data.hallucination_rate !== undefined
            ? data.hallucination_rate
            : "n/a"}
        </strong>
      </div>

      {/* Latency */}
      <div className="metric">
        <span>Latency (seconds)</span>
        <strong>
          Avg: {latency.avg ?? "n/a"} | Min: {latency.min ?? "n/a"} | Max:{" "}
          {latency.max ?? "n/a"}
        </strong>
      </div>

      {/* Embedding Similarity */}
      <div className="metric">
        <span>Embedding Similarity (Cosine)</span>
        <strong>
          Mean: {embedding.mean ?? "n/a"}
        </strong>
      </div>

      {/* Retrieval Precision */}
      <div className="metric">
        <span>Retrieval Precision</span>
        <strong>{data.retrieval_precision ?? "n/a"}</strong>
      </div>

      {/* Retrieval Recall */}
      <div className="metric">
        <span>Retrieval Recall</span>
        <strong>{data.retrieval_recall ?? "n/a"}</strong>
      </div>

      {/* Questions Evaluated */}
      <div className="metric">
        <span>Questions Evaluated</span>
        <strong>{data.num_questions ?? "n/a"}</strong>
      </div>
    </div>
  );
}
