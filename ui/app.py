# ui/app.py
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from ingestion.ingestion import extract_and_chunk_docs, create_vectorstore
from agents.supervisor_agent import invoke_supervisor
import shutil

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "chromadb_store")

# In-memory cached vectordb + docs for quick testing (persisted on disk via Chroma)
_vectordb = None
_docs = None

@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "status": "uploaded"}

@app.post("/ingest/")
async def ingest_all():
    # ingest all files present in UPLOAD_DIR
    paths = [os.path.join(UPLOAD_DIR, p) for p in os.listdir(UPLOAD_DIR)]
    if not paths:
        return JSONResponse(content={"error": "no files uploaded"}, status_code=400)
    docs = extract_and_chunk_docs(paths)
    vectordb = create_vectorstore(docs, persist_directory=CHROMA_DIR)
    global _vectordb, _docs
    _vectordb = vectordb
    _docs = docs
    return {"status": "ingested", "chunks": len(docs)}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global _vectordb, _docs
    if _vectordb is None or _docs is None:
        return JSONResponse(content={"error": "no vectorstore, please call /ingest/ first"}, status_code=400)
    result = invoke_supervisor(question, vectorstore=_vectordb, docs=_docs)
    # answer may be a runnable object or raw model output — return as-is
    return JSONResponse(content=result)

@app.get("/ui", response_class=HTMLResponse)
async def frontend():
    # serve a minimal HTML page or point to Streamlit UI; here we return a link
    return "<html><body><h3>GenAI Analyst</h3><p>Use the Streamlit UI at /streamlit (separate)</p></body></html>"
