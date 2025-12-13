import React, {useState} from 'react';
import axios from 'axios';


export default function Query(){
  const [q,setQ]=useState('');
  const [res,setRes]=useState(null);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState(null);

  const ask = async ()=>{
    setError(null);
    setRes(null);
    if(!q || q.trim().length===0) return setError('Enter a question');
    setLoading(true);
    try {
      const form = new FormData();
      form.append('question', q);
      const r = await axios.post('http://localhost:8000/query', form, { timeout: 60000 });
      setRes(r.data);
    } catch (err) {
      console.error(err);
      if(err.response){
        // server returned an error
        setError(`Server error ${err.response.status}: ${JSON.stringify(err.response.data)}`);
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

return (
  <div className="section">
    <h2>🔍 Ask a Question</h2>

    <input
      type="search"
      className="input-box"
      placeholder="Ask analyst anything..."
      value={q}
      onChange={(e) => setQ(e.target.value)}
    />

    <button onClick={ask} disabled={loading}>
      {loading ? "Thinking..." : "Ask"}
    </button>

    {error && (
      <div style={{ color: "red", marginTop: 10, whiteSpace: "pre-wrap" }}>
        {error}
      </div>
    )}

    {res && (
      <>
        <h3 style={{ marginTop: 20 }}>Answer</h3>
        <pre className="pre-box">
          {JSON.stringify(res.answer || res, null, 2)}
        </pre>

        <h3>Retrieval Diagnostics</h3>
        <pre className="pre-box" style={{ background: "#fff7e6" }}>
          {JSON.stringify(res.retrieval?.diagnostics || {}, null, 2)}
        </pre>

        <h3>Top Chunks</h3>
        {res.retrieval?.chunks?.map((chunk, idx) => (
          <div className="chunk-card" key={idx}>
            <strong>ID:</strong> {chunk.id} <br />
            <strong>Score:</strong> {chunk.score}
            <div className="chunk-text">
              {chunk.metadata?.text?.slice(0, 800)}
            </div>
          </div>
        ))}
      </>
    )}
  </div>
);


}

