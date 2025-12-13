import React from "react";

export default function Evaluation({ data }) {
  if (!data) return null;

  return (
    <div className="evaluation-card">
      <h2>📊 System Evaluation</h2>

      <div className="metric">
        <span>Hallucination Rate</span>
        <strong>{data.hallucination_rate}</strong>
      </div>

      <div className="metric">
        <span>Latency (seconds)</span>
        <strong>
          Avg: {data.latency.avg} | Min: {data.latency.min} | Max: {data.latency.max}
        </strong>
      </div>

      <div className="metric">
        <span>Embedding Similarity (Cosine)</span>
        <strong>
          Mean: {data.embedding_similarity.mean}
        </strong>
      </div>

      <div className="metric">
        <span>Questions Evaluated</span>
        <strong>{data.num_questions}</strong>
      </div>
    </div>
  );
}

