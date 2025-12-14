# agents/answer_agent.py
import os
from dotenv import load_dotenv
load_dotenv()
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
from prompts.answer_generation_prompt import ANSWER_GENERATION_PROMPT

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

answer_agent = create_deep_agent(
    tools=[],
    system_prompt=ANSWER_GENERATION_PROMPT,
    model=model,
)
