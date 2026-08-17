"""Конфигурация корпоративного ассистента."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Загружаем .env из корня проекта
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

PROJECT_ROOT = _PROJECT_ROOT

RAW_PDFS = PROJECT_ROOT / "data" / "raw" / "pdfs"
RAW_WIKI = PROJECT_ROOT / "data" / "raw" / "wiki"
CHUNKS_JSON = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
RESULTS_DIR = PROJECT_ROOT / "results"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Модель эмбеддингов: локальная мультиязычная sentence-transformers.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = 384  # размерность multilingual-e5-small

# Yandex Cloud LLM
YC_API_KEY = os.getenv("YC_API_KEY", "")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID", "")
YC_URL = os.getenv(
    "YC_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)
YC_MODEL = os.getenv("YC_MODEL", "yandexgpt-lite")

# Локальная LLM через Ollama (см. `ollama list`).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE = 0.0

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "corporate_assistant_2026")

TOP_K = 5
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60
