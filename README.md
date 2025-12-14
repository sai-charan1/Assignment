# GenAI Analyst - AI Document Analyst System

This project implements an internal AI analyst system to answer complex queries on uploaded policy documents, product manuals, and financial reports. It uses modular agents orchestrated via LangChain deepagents and Azure OpenAI API, with a retrieval-augmented generation (RAG) pipeline for document ingestion and hybrid retrieval.

---

## Features

- Document Type Classification with chain-of-thought and JSON schema outputs
- Advanced Answer Generation with citations, confidence scoring, and reasoning
- Summarization optimized for RAG embeddings
- RAG pipeline with semantic chunking, vector embeddings (Chroma), keyword BM25, and re-ranking
- Multi-agent workflow with Query Analyzer, Retrieval, and Answering Agents coordinated by a Supervisor
- Evaluation scripts to measure hallucination rate, retrieval metrics, and latency
- Web UI for uploading documents and querying the AI analyst
- Dockerized for easy deployment

---

## Folder Structure

genai-agent-project/
│
├── prompts/ # Prompt templates for classification, answer generation, summarization
├── rag_pipeline/ # Document ingestion and retrieval logic
├── agents/ # Deepagents for multi-agent orchestration
├── evaluation/ # Evaluation scripts and test datasets
├── ui/ # FastAPI web app to upload docs and ask queries
├── Dockerfile # Containerization file
├── requirements.txt # Python dependencies
└── README.md # This file

text

---

## Setup and Installation

1. Clone the repository:
git clone <repo-url>
cd genai-agent-project

text

2. Set up a Python virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate

text

3. Install dependencies:
pip install -r requirements.txt

text

4. Set environment variables for API keys (`.env` or shell):
export AZURE_OPENAI_API_KEY="your_azure_openai_key"
export TAVILY_API_KEY="your_tavily_key"

text

5. Update deployment names and paths in the code where needed.

---

## Running the System

### Start the API Server
uvicorn ui.app:app --host 0.0.0.0 --port 8000 --reload

text

- Upload documents via POST `/upload_document/`
- Ask questions via POST `/ask/`
- Visit `http://localhost:8000` for welcome message

### Run Evaluation
python evaluation/evaluation_script.py

text

---

## Module Overview

### Prompts
Defines task-specific prompts with chain-of-thought, JSON outputs, and fallback handling.

### RAG Pipeline
Extracts text from PDFs/text, semantically chunks, generates embeddings, stores in Chroma, and implements hybrid retrieval.

### Agents
GenAI analyst workflow split into Query Analyzer, Retrieval Agent, Answer Agent, coordinated by Supervisor agent using deepagents framework.

### Evaluation
Scripts to benchmark hallucination rate, precision/recall on retrieval, latency, and embedding quality.

### UI
Lightweight FastAPI app for document management and interactive querying.

---



