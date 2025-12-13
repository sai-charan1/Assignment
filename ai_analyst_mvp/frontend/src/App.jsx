import React from "react";
import Upload from "./components/Upload";
import Query from "./components/Query";
import Results from "./components/Results";
import Evaluation from "./components/Evaluation";
import "./styles.css";

export default function App() {
  const [response, setResponse] = React.useState(null);
  const [topChunks, setTopChunks] = React.useState(null);
  const [evaluation, setEvaluation] = React.useState(null);
  const [isEvaluating, setIsEvaluating] = React.useState(false); // ✅ added

  return (
    <div className="main-bg">
      <h1 className="heading">AI Analyst MVP</h1>

      <div className="content-wrapper">
        {/* Upload */}
        <div className="section">
          <Upload />
        </div>

        {/* Query */}
        <div className="section">
          <Query setResponse={setResponse} setTopChunks={setTopChunks} />
        </div>

        {/* Results */}
        {response && (
          <div className="section">
            <Results response={response} topChunks={topChunks} />
          </div>
        )}

        {/* Evaluation */}
        <div className="section">
          <button
            disabled={isEvaluating}
            onClick={async () => {
              setIsEvaluating(true);
              setEvaluation(null);

              try {
                const res = await fetch("http://localhost:8000/evaluate");
                const data = await res.json();
                setEvaluation(data);
              } catch (err) {
                console.error("Evaluation failed", err);
              } finally {
                setIsEvaluating(false);
              }
            }}
          >
            {isEvaluating ? "Running Evaluation..." : "Run Evaluation"}
          </button>

          {isEvaluating && (
            <div style={{ marginTop: "12px", fontWeight: 500, color: "#4f46e5" }}>
              ⏳ Running system evaluation, please wait...
            </div>
          )}

          <Evaluation data={evaluation} />
        </div>
      </div>
    </div>
  );
}

