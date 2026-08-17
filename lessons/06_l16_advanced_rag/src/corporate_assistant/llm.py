"""Инициализация LLM через Ollama."""

from __future__ import annotations

from llama_index.llms.ollama import Ollama

from corporate_assistant.config import LLM_TEMPERATURE


def build_llm(model: str, base_url: str) -> Ollama:
    return Ollama(
        model=model,
        base_url=base_url,
        temperature=LLM_TEMPERATURE,
        request_timeout=600.0,
    )
