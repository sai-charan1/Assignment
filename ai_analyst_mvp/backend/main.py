# backend/main.py (replace contents)
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from .utils import ensure_dir
import shutil, uuid, logging, time
from dotenv import load_dotenv
from pathlib import Path
from .evaluate import run_evaluation

import os

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title='AI Analyst MVP')

# Allow common localhost origins used by browsers/dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent.parent / 'uploads'
ensure_dir(UPLOAD_DIR)

# Agents are initialized at startup to avoid heavy imports on module import
analyzer = None
retriever = None
answer_agent = None
supervisor = None  # optional if you later implement a SupervisorAgent

@app.on_event("startup")
def startup_event():
    global analyzer, retriever, answer_agent, supervisor
    try:
        from .agents import QueryAnalyzerAgent, RetrievalAgent, AnswerAgent
        # SupervisorAgent is optional — safe import if present
        try:
            from .agents import SupervisorAgent
        except Exception:
            SupervisorAgent = None
        analyzer = QueryAnalyzerAgent()
        retriever = RetrievalAgent()
        answer_agent = AnswerAgent()
        if SupervisorAgent is not None:
            try:
                supervisor = SupervisorAgent(analyzer=analyzer, retriever=retriever, answer_agent=answer_agent)
                logger.info("SupervisorAgent initialized.")
            except Exception as e:
                supervisor = None
                logger.warning("SupervisorAgent present but failed to initialize: %s", e)
        logger.info("Agents initialized on startup.")
    except Exception as e:
        logger.exception("Failed to initialize agents at startup: %s", e)
        # keep server up; queries will fail more gracefully later

@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    """
    Upload endpoint (multipart/form-data). Saves file under uploads/ and triggers ingestion.
    Returns JSON with chunks_created and filename.
    """
    try:
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
        # write file
        with open(dest, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Saved upload to %s", dest)

        # import ingest_file lazily (so startup/import stays cheap)
        from .ingestion import ingest_file
        start = time.time()
        count = ingest_file(dest, doc_id=dest.stem)
        took = time.time() - start
        logger.info("Ingested %s -> %d chunks (%.2fs)", dest.name, count, took)
        return JSONResponse({'status': 'ok', 'chunks_created': count, 'filename': dest.name, 'ingest_time_s': took})
    except Exception as e:
        logger.exception("Upload/ingest failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post('/query')
async def query_endpoint(request: Request, question: str = Form(None)):
    """
    Query endpoint accepts either:
    - form field `question` (used by simple form POST),
    - or JSON body { "question": "..." } or { "query": "..." } used by frontend code.
    Returns plan, retrieval (chunks + diagnostics + time), and answer.
    """
    if analyzer is None or retriever is None or answer_agent is None:
        raise HTTPException(status_code=503, detail="Agents not initialized yet. Please retry after server startup completes.")

    try:
        # Accept form or JSON
        if not question:
            try:
                body = await request.json()
            except Exception:
                body = {}
            question = body.get("question") or body.get("query") or body.get("q")
        if not question:
            raise HTTPException(status_code=400, detail="No question provided in 'question' form field or JSON body.")

        total_start = time.time()

        # Optionally use a SupervisorAgent if available to orchestrate full workflow
        if supervisor is not None:
            try:
                response = supervisor.handle_question(question)
                response.setdefault("time_total_s", time.time() - total_start)
                return JSONResponse(response)
            except Exception as e:
                logger.exception("SupervisorAgent failed; falling back to manual pipeline: %s", e)

        # 1) plan
        plan = analyzer.analyze(question)

        # 2) retrieval
        r_start = time.time()
        retrieval_result = retriever.execute(plan)
        r_time = time.time() - r_start

        chunks = retrieval_result.get('chunks', [])

        # 3) answer
        a_start = time.time()
        ans = answer_agent.answer(question, chunks)
        a_time = time.time() - a_start

        total_time = time.time() - total_start

        # structured response (helpful for frontend + evaluation)
        response = {
            "plan": plan,
            "retrieval": {
                "chunks": chunks,
                "diagnostics": retrieval_result.get("diagnostics", {}),
                "time_s": retrieval_result.get("time", r_time)
            },
            "answer": ans,
            "timing": {
                "retrieval_s": r_time,
                "answer_s": a_time,
                "total_s": total_time
            }
        }
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query handling failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@app.get("/evaluate")
def evaluate_endpoint():
    """
    Runs system evaluation and returns metrics for the frontend.
    """
    try:
        result = run_evaluation()

        # ensure frontend-safe schema
        result.setdefault("latency", {})
        result.setdefault("embedding_similarity", {})
        result.setdefault("retrieval_pr", {})
        result.setdefault("hallucination_rate", None)

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return JSONResponse(
            {
                "error": str(e),
                "latency": {},
                "embedding_similarity": {},
                "retrieval_pr": {},
                "hallucination_rate": None,
            },
            status_code=500,
        )
