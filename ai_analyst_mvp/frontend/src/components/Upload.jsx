import React, { useState } from "react";
import axios from "axios";
import "../styles.css";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const upload = async () => {
    if (!file) return setMsg("Choose a file first");
    setMsg("");
    setLoading(true);
    setProgress(0);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await axios.post("http://127.0.0.1:8000/upload", fd, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (ev) => {
          if (ev.lengthComputable) setProgress(Math.round((ev.loaded / ev.total) * 100));
        },
        timeout: 10 * 60 * 1000,
      });
      setMsg(`Upload succeeded:\n${JSON.stringify(res.data, null, 2)}`);
    } catch (err) {
      console.error("Upload error", err);
      if (err.response) {
        setMsg(`Server error: ${err.response.status} ${err.response.data?.detail || ""}`);
      } else if (err.request) {
        setMsg("Network error: request sent but no response (CORS/Network/Timeout).");
      } else {
        setMsg(`Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

 return (
  <div className="section">
    <h2>📄 Upload PDF</h2>

    <input
      type="file"
      accept=".pdf"
      onChange={(e) => {
        setFile(e.target.files[0]);
        setMsg("");
      }}
    />

    <button onClick={upload} disabled={loading} style={{ marginTop: 10 }}>
      {loading ? `Uploading` : "Upload"}
    </button>

    

    {msg && <pre className="pre-box">{msg}</pre>}
  </div>
);


}
