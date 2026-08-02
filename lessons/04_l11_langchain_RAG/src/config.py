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
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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

SYSTEM_PROMPT_TEMPLATE = """Ты корпоративный ассистент. Отвечай на вопрос пользователя \
ТОЛЬКО опираясь на следующий контекст, полученный из внутренних документов компании.
Если ответа в контексте нет — прямо скажи, что не знаешь, и не придумывай факты.
Отвечай сразу, без внутренних рассуждений. /no_think

После ответа ОБЯЗАТЕЛЬНО добавь строку в формате:
ИСПОЛЬЗОВАНО: <название файла>, стр. <номер> (перечисли только те источники из контекста,
на которые ты реально опирался при ответе; если использовал несколько - перечисли через запятую)

Контекст:
{context}
"""
