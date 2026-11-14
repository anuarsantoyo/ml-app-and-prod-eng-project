"""OpenRouter chat client (shared helper)."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


def model_name() -> str:
    return os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
