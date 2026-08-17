"""Сборка векторного индекса: парсинг документов -> чанки с метаданными -> Qdrant.

Запуск:
  uv run python scripts/build_index.py            # сборка, если коллекции нет
  uv run python scripts/build_index.py --reset    # пересобрать с нуля
"""

from __future__ import annotations

import argparse

from corporate_assistant.config import COLLECTION_NAME
from corporate_assistant.indexing import build_index, qdrant_client
from corporate_assistant.parsing import load_all_chunks, save_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Парсинг документов и сборка Qdrant-индекса.")
    parser.add_argument("--reset", action="store_true", help="Пересоздать коллекцию с нуля.")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()

    client = qdrant_client()
    exists = client.collection_exists(args.collection)
    if exists and not args.reset:
        print(f"Коллекция {args.collection!r} уже существует. Используйте --reset для пересборки.")
        return 0

    print("Парсинг документов и нарезка чанков...")
    chunks = load_all_chunks()
    print(f"Всего чанков: {len(chunks)}")
    save_chunks(chunks)
    print("Чанки сохранены в data/chunks/chunks.json")

    build_index(chunks, reset=args.reset, collection_name=args.collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
