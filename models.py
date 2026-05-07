"""
Lazy LangChain ChatModel objects for the model pool.

Usage:
    from models import MODELS

    response = MODELS["gpt-4o-mini"].invoke("Hello")

Provider API keys are needed only when a model object is first constructed.
"""

from collections.abc import Mapping

from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODEL_SPECS = {
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4o": ("openai", "gpt-4o"),
    "llama-3.1-8b": ("together", "meta-llama/Meta-Llama-3-8B-Instruct-Lite"),
    "llama-3.3-70b": ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "qwen2.5-7b": ("together", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    "deepseek-v3": ("together", "deepseek-ai/DeepSeek-V3.1"),
    "gpt-oss-20b": ("together", "openai/gpt-oss-20b"),
    "MiniMax-M2.7": ("together", "MiniMaxAI/MiniMax-M2.7"),
}


def build_model(name: str):
    provider, model_id = MODEL_SPECS[name]
    if provider == "openai":
        return ChatOpenAI(model=model_id)
    if provider == "together":
        return ChatTogether(model=model_id)
    raise ValueError(f"Unknown provider for {name}: {provider}")


class LazyModelRegistry(Mapping):
    def __init__(self, specs: dict[str, tuple[str, str]]):
        self._specs = specs
        self._cache = {}

    def __getitem__(self, key: str):
        if key not in self._cache:
            self._cache[key] = build_model(key)
        return self._cache[key]

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)


MODELS = LazyModelRegistry(MODEL_SPECS)

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
            print(f"ERROR: {e}")
