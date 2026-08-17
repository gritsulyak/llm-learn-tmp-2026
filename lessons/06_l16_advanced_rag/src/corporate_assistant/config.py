"""Конфигурация корпоративного ассистента."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PDFS = PROJECT_ROOT / "data" / "raw" / "pdfs"
RAW_WIKI = PROJECT_ROOT / "data" / "raw" / "wiki"
CHUNKS_JSON = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
RESULTS_DIR = PROJECT_ROOT / "results"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Модель эмбеддингов: локальная мультиязычная sentence-transformers.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = 384  # размерность multilingual-e5-small

# Локальная LLM через Ollama (см. `ollama list`).
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE = 0.0

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "corporate_assistant_2026")

TOP_K = 5
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60
