# AGENTS.md

## Project Overview

Corporate Knowledge Assistant — продвинутый RAG-пайплайн на LlamaIndex + Qdrant + Ollama. Ищет ответы в PDF-инструкциях и Wiki-статьях, отвечает строго по контексту с цитированием источников.

## Stack

- Python 3.12+, managed with `uv`
- LlamaIndex (RAG framework)
- Qdrant (vector DB, via Docker)
- Ollama (`qwen2.5:3b` for LLM and judge)
- `intfloat/multilingual-e5-small` embeddings (CPU)
- PyMuPDF for PDF parsing, BeautifulSoup for HTML/Wiki parsing

## Key Commands

```bash
# Lint / format (if configured)
uv run ruff check src/ scripts/ tests/
uv run ruff format src/ scripts/ tests/

# Unit tests
uv run pytest -q

# Integration tests (requires Qdrant + Ollama running)
docker compose up -d qdrant
uv run python scripts/build_index.py --reset
uv run python scripts/run_tests.py
uv run python scripts/evaluate_ragas.py
```

## Project Structure

```
src/corporate_assistant/
  config.py        Constants (URL, models, chunk sizes)
  parsing.py       PyMuPDF + bs4: text/metadata extraction, chunking
  indexing.py       Embeddings, Qdrant index build/read, retriever
  prompts.py       System prompt, context formatting, grounding
  assistant.py     RAG pipeline (retrieval → LLM → citation check)
  llm.py           Ollama wrapper
scripts/           CLI scripts (build index, run tests, chat, evaluate)
tests/test_core.py Pytest: parser, chunking, formatting, grounding
```

## Code Conventions

- Package source lives under `src/corporate_assistant/` (hatch build layout)
- Tests go in `tests/test_core.py`
- Scripts in `scripts/` are entry points, not library code
- All text/comments in source are in Russian (project is Russian-language)
- Config constants centralized in `src/corporate_assistant/config.py`
- No external API keys needed — everything runs locally (Ollama + Qdrant)

## Before Committing

1. Run `uv run pytest -q` — unit tests must pass
2. Run `uv run ruff check` — no lint errors
3. Ensure imports resolve with the hatch `src` layout
