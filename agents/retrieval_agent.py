# agents/retrieval_agent.py
import os
from dotenv import load_dotenv
load_dotenv()
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI

from ingestion.retrieval import HybridRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# Model for any small text manipulation steps (not the retrieval core)
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

retrieval_agent_prompt = """
Retrieval agent: receives a plan with 'query', 'retrieval_strategy', and 'top_k'.
Return JSON:
{
  "top_chunks": [ {"source":"...","text":"...","score":0.0}, ... ],
  "contradictions": "<if any>",
  "diagnostics": { ... }
}
"""

# We create the deep agent that wraps model for small rewriting / decision logic
retrieval_agent = create_deep_agent(
    tools=[],  # actual retrieval executed externally by orchestrator
    system_prompt=retrieval_agent_prompt,
    model=model,
)
