import React from 'react';
export default function Results({data}) {
  if(!data) return null;
return (
  <div className="section">
    <h2>📘 Answer</h2>

    <div className="pre-box">
      {data.answer?.answer}
    </div>

    <h3>📑 Evidence</h3>
    <ul>
      {data.answer?.evidence?.map((e, i) => (
        <li key={i}>
          <strong>{e.source}</strong> [{e.chunk_index}] — {e.text}
        </li>
      ))}
    </ul>
  </div>
);

}
