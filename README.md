# AI_Analyst

End-to-End Retrieval-Augmented AI Analyst for Complex Documents

**📌 Problem Statement**

This project implements an internal AI Analyst MVP for a consulting firm.
The system enables analysts to upload large documents (policies, manuals, financial reports) and ask complex, evidence-backed questions.

The system is designed to:

Understand different document types

Retrieve most relevant supporting evidence

Perform reasoning and synthesis

Produce structured, explainable outputs

Use a multi-agent workflow to orchestrate tools

Provide a usable UI for real analysts


**🧩 Architecture Overview**

Pipeline Flow:

        User Uploads PDF
                ↓
        Semantic Ingestion & Chunking
                ↓
        Embedding + Metadata Storage (ChromaDB)
                ↓
        Query → Query Analyzer Agent
                ↓
        Retrieval Agent (Hybrid Search)
                ↓
        Answer Agent (Reasoning + Citations)
                ↓
        Structured JSON Response
                ↓
        Frontend UI Rendering
        

# RAG System Implementation

**A. Ingestion**

PDF text extraction

Semantic chunking (not fixed-size)

Metadata preserved (source, page, section)

Embeddings stored in ChromaDB

**B. Retrieval (Hybrid)**

Three-stage retrieval:

Vector similarity (semantic)

BM25 keyword matching

Cross-ranking merge

**System Outputs:**

Top 5 ranked chunks

Retrieval diagnostics

Scores per strategy      


# Multi-Agent Workflow

Coordinates all agents and ensures task completion.

**Agent 1: Query Analyzer Agent**

Determines user intent (factual, reasoning, comparison, multi-hop)

Decides retrieval strategy

Rewrites queries if needed

Produces execution plan


**Agent 2: Retrieval Agent**

Executes hybrid retrieval

Ranks and filters chunks

Surfaces contradictory evidence

**Agent 3: Answer Agent**

Applies Answer Generation Prompt

Produces structured output

Responds as a domain analyst, not a chatbot


# 🧱 Tech Stack
**Backend** 

FastAPI (Python)

Azure OpenAI API (LLM)

ChromaDB (vector store)

Sentence Transformers (MiniLM-L6-v2)

BM25 (rank-bm25)

Python-dotenv

Uvicorn

**Frontend** 

React



# 📂 Project Structure    


      ai_analyst_mvp/
      │
      ├── backend/
      │   ├── main.py
      │   ├── agents.py
      │   ├── retriever.py
      │   ├── ingestion.py
      │   ├── llm_tools.py
      │   ├── prompts/
      │   │   ├── classifier_prompt.txt
      │   │   ├── answer_prompt.txt
      │   │   ├── summarization_prompt.txt
      │   │   └── hidden_instruction.txt
      │   ├── .env  (NOT COMMITTED)
      │   ├── requirements.txt
      │   └── ...
      │
      ├── frontend/
      │   ├── public/index.html
      │   ├── src/
      │   │   ├── App.jsx
      │   │   ├── styles.css
      │   │   ├── components/
      │   │   │   ├── Upload.jsx
      │   │   │   ├── Query.jsx
      │   │   │   └── Results.jsx
      │   ├── package.json
      │
      ├── .gitignore
      └── README.md
      


# 🔐 Environment Variables

Create a .env file inside /backend:

          AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
          AZURE_OPENAI_API_KEY=<your-key>
          AZURE_OPENAI_API_VERSION=2024-08-01-preview
          AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini



# ⚙️ Backend Setup
1️⃣ Activate virtual environment

      cd backend
      python3 -m venv .venv
      source .venv/bin/activate

2️⃣ Install dependencies

    pip install -r requirements.txt

3️⃣ Run the backend server

    uvicorn main:app --reload --port 8000

Backend runs at:

      http://127.0.0.1:8000


# 🌐 Frontend Setup
1️⃣ Install packages

    cd frontend
    npm install

2️⃣ Run React app

    npm start

Frontend runs at:

    http://localhost:3000
