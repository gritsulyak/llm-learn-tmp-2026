"""
Конфигурация проекта. Все параметры читаются из .env (или переменных окружения).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3:latest")

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "deepvk/USER-bge-m3-v1",
)

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_docs")
QDRANT_TEST_COLLECTION = os.getenv("QDRANT_TEST_COLLECTION", "enterprise_docs_test")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
TOP_K = int(os.getenv("TOP_K", 4))

# --- Paths ---
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "docs"))
TEST_DOCS_DIR = Path(os.getenv("TEST_DOCS_DIR", BASE_DIR / "docs_corpus_sample"))

SYSTEM_PROMPT_TEMPLATE = PROMPT_TEMPLATE = """Ты — ассистент, который отвечает строго по контексту.

Правила:
1. Используй ТОЛЬКО текст внутри <context>...</context>.
2. Если в контексте нет ответа на вопрос, верни ровно: ответа не найдено
3. Не добавляй ничего от себя.

Формат ответа всегда:
ответ: <текст ответа или ответа не найдено>
ИСПОЛЬЗОВАНО: <название источника из квадратных скобок>

Если ответ не найден, пиши:
ИСПОЛЬЗОВАНО: нет

Пример 1:
<context>
[Источник: газета.txt]
малиновое варенье готовится из малины
---
[Источник: книжка2.pdf]
чай пьют по утрам
</context>
Вопрос: напиши рецепт чая.

ответ: ответа не найдено
ИСПОЛЬЗОВАНО: нет

Пример 2:
<context>
[Источник: мураши.doc]
муравьи живут ровно там где мухи
---
[Источник: мухи.pdf]
мухи живут там где тепло и солнечно
</context>
Вопрос: Где живут муравьи?

ответ: муравьи живут ровно там где мухи
ИСПОЛЬЗОВАНО: мураши.doc, мухи.pdf

<context>
{context}
</context>
"""
