"""Интерактивный чат с корпоративным ассистентом.

Каждый запрос проходит RAG-пайплайн: retrieval -> контекст -> LLM -> grounding.

Запуск: uv run python scripts/chat.py [--collection corporate_assistant_2026] [--top-k 5]
Выход: пустая строка, "exit", "quit" или Ctrl+D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from corporate_assistant.assistant import CorporateAssistant
from corporate_assistant.config import COLLECTION_NAME, TOP_K


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    assistant = CorporateAssistant(collection_name=args.collection, top_k=args.top_k)
    print(f"Корпоративный ассистент АО «Осинка» ({args.collection}, top_k={args.top_k})")
    print("Задавайте вопросы. Пустая строка / exit / Ctrl+D — выход.\n")

    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            print()
            break
        if not query or query.lower() in {"exit", "quit"}:
            break

        result = assistant.answer(query)
        print(result.answer)
        if result.dropped_citations:
            print(f"[удалены выдуманные ссылки: {', '.join(result.dropped_citations)}]")
        if result.sources:
            print("Источники:")
            for s in result.sources:
                meta = s["metadata"]
                label = meta.get("source", "?")
                section = meta.get("header", "")
                page = meta.get("page")
                if meta.get("doc_type") == "wiki":
                    print(f"  • {label} — раздел «{section}»")
                else:
                    print(f"  • {label} — раздел «{section}», стр. {page}")
        print()


if __name__ == "__main__":
    main()