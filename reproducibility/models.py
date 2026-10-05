"""
LangChain ChatModel objects for all model pairs.

Usage:
    from models import MODELS, cheap_models, expensive_models

    response = MODELS["gpt-4o-mini"].invoke("Hello")

Load environment variables before importing:
    uv run --env-file .env python your_script.py
"""

from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = {
    # --- OpenAI ---
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
    "gpt-4o":      ChatOpenAI(model="gpt-4o"),

    # --- Llama via Together ---
    "llama-3.1-8b":  ChatTogether(model="meta-llama/Meta-Llama-3-8B-Instruct-Lite"),
    "llama-3.3-70b": ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),

    # --- Qwen via Together ---
    "qwen2.5-7b": ChatTogether(model="Qwen/Qwen2.5-7B-Instruct-Turbo"),

    # --- DeepSeek via Together ---
    "deepseek-v3": ChatTogether(model="deepseek-ai/DeepSeek-V3.1"),

    # --- GPT-OSS-20B via Together ---
    "gpt-oss-20b": ChatTogether(model="openai/gpt-oss-20b"),

    # --- MiniMax via Together ---
    "MiniMax-M2.7": ChatTogether(model="MiniMaxAI/MiniMax-M2.7"),
}

CHEAP_MODELS     = ["gpt-4o-mini", "llama-3.1-8b",  "qwen2.5-7b"]
EXPENSIVE_MODELS = ["gpt-4o",      "llama-3.3-70b", "deepseek-v3"]

PAIRS = list(zip(CHEAP_MODELS, EXPENSIVE_MODELS))  # [(cheap, expensive), ...]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_prompt = "Reply with one word: hello."

    for name, model in MODELS.items():
        print(f"{name}: ", end="", flush=True)
        try:
            response = model.invoke(test_prompt)
            print(response.content.strip())
        except Exception as e:
            print(f"ERROR — {e}")
