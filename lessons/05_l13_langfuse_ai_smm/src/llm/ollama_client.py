from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import settings


@lru_cache(maxsize=None)
def get_llm(model: str, temperature: float | None = None) -> ChatOllama:
    """Cached ChatOllama client per (model, temperature) pair."""
    cfg = settings()
    return ChatOllama(
        model=model,
        base_url=cfg.ollama_base_url,
        temperature=temperature if temperature is not None else cfg.llm_temperature,
        timeout=cfg.llm_request_timeout_s,
    )
