# agents/query_analyzer_agent.py
import os
from dotenv import load_dotenv
load_dotenv()
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
from prompts.document_type_classifier_prompt import DOCUMENT_TYPE_CLASSIFIER_PROMPT

# Model (independent inside each agent)
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

query_analyzer_prompt = """
You are a Query Analyzer Agent for a document-grounded AI system.

Your job is to analyze the user's question and decide HOW retrieval should be done.
You MUST NOT change the topic of the user's question.

STRICT RULES:
- NEVER introduce a new topic or domain
- NEVER hallucinate or generalize beyond the user question
- NEVER replace the question with an unrelated example
- If unsure, reuse the original question verbatim

Tasks:
1. Classify the question intent
2. Decide the best retrieval strategy
3. Decide how many chunks to retrieve
4. Return a retrieval plan ONLY

Intent types:
- factual
- reasoning
- comparison
- multi-hop
- missing_data

Retrieval strategy rules:
- procedural / how-to questions → hybrid
- exact definitions → vector
- keyword-heavy questions → bm25

Output STRICT JSON ONLY:
{
  "intent": "<intent>",
  "retrieval_strategy": "<vector|bm25|hybrid>",
  "top_k": 5,
  "query": "<EXACT user question, unchanged>"
}
"""


# build agent
query_analyzer_agent = create_deep_agent(
    tools=[],  # no external tools; it produces a plan
    system_prompt=query_analyzer_prompt,
    model=model,
)
