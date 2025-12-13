# backend/llm_tools.py

from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

# ================= LOAD ENV ==================
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")   # e.g., gpt-4o-mini
API_VERSION = "2024-08-01-preview"

# ================= CLIENT =====================
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    api_version=API_VERSION,
)

# ================= LLM CALL FUNCTION ==========
def call_llm(prompt: str, max_tokens: int = 800) -> str:
    print("🔥 Azure OpenAI LLM WAS CALLED 🔥")

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,     # Your Azure deployment name
            messages=[
                {"role": "system", "content": "You are an AI analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {str(e)}"

