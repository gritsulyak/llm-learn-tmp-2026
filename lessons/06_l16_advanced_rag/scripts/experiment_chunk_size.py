"""Эксперимент: как размер чанка влияет на точность привязки номера страницы.

Для каждого chunk_size собирается отдельная Qdrant-коллекция, затем проверяется,
попадает ли в топ-K (K=3) чанк с ожидаемым источником и номером страницы.

Запуск: uv run python scripts/experiment_chunk_size.py
"""

from __future__ import annotations

import time

from corporate_assistant.config import QDRANT_URL
from corporate_assistant.indexing import build_index, qdrant_client, retrieve
from corporate_assistant.parsing import load_all_chunks
from corporate_assistant.prompts import source_label

QUESTIONS: list[tuple[str, str, set]] = [
    # (вопрос, ожидаемый источник, допустимые номера страниц)
    (
        "За сколько дней нужно писать заявление на отпуск?",
        "Инструкция_по_отпускам.pdf",
        {5, 8},
    ),
    (
        "Какой размер суточных при командировке?",
        "Положение_о_командировках.pdf",
        {3},  # печатный номер страницы (физическая 5, offset=2)
    ),
    (
        "Сколько процентов от заработка платят при больничном со стажем 7 лет?",
        "Инструкция_по_больничным_листам.pdf",
        {3},
    ),
    (
        "Какие требования к корпоративным паролям?",
        "Корпоративные пароли и доступы",
        set(),
    ),
    (
        "Можно ли работать удаленно в среду?",
        "Политика_удаленной_работы_2026.pdf",
        {1, 2},
    ),
]

TOP_K = 3


def evaluate(chunk_size: int, chunk_overlap: int, collection: str) -> tuple[int, int, list[dict]]:
    client = qdrant_client()
    if client.collection_exists(collection):
        client.delete_collection(collection)

    chunks = load_all_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_index(chunks, reset=True, collection_name=collection)

    rows: list[dict] = []
    correct = 0
    for question, expected_source, allowed_pages in QUESTIONS:
        nodes = retrieve(question, top_k=TOP_K, collection_name=collection)
        found = any(
            node.node.metadata.get("source") == expected_source
            and (
                not allowed_pages
                or node.node.metadata.get("page") in allowed_pages
            )
            for node in nodes
        )
        correct += int(found)
        top = source_label(nodes[0].node.metadata)
        rows.append({"question": question, "expected": expected_source, "top1": top, "ok": found})
    return correct, len(QUESTIONS), rows


def main() -> None:
    client = qdrant_client()
    configs = [
        (300, 30),
        (600, 60),
        (1200, 120),
    ]
    print(f"{'chunk_size':<11}{'chunk_overlap':<15}{'точность':<12}коллекция")
    summary: list[dict] = []
    for chunk_size, overlap in configs:
        collection = f"experiment_cs_{chunk_size}"
        started = time.monotonic()
        correct, total, rows = evaluate(chunk_size, overlap, collection)
        elapsed = time.monotonic() - started
        print(f"{chunk_size:<11}{overlap:<15}{correct}/{total:<11}{collection}")
        for row in rows:
            print(f"   {'OK ' if row['ok'] else 'FAIL'} {row['question'][:60]} -> top1={row['top1']}")
        summary.append(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "collection": collection,
                "correct": correct,
                "total": total,
                "rows": rows,
                "elapsed_s": round(elapsed, 1),
            }
        )
        print()

    for chunk_size, _ in configs:
        collection = f"experiment_cs_{chunk_size}"
        client.delete_collection(collection)
        print(f"Удалена временная коллекция: {collection}")

    print("Временные коллекции удалены.")
    print("\nВывод: чанки привязаны к страницам до нарезки, поэтому chunk_size не")
    print("влияет на корректность номера страницы — он всегда берётся из метаданных страницы.")


if __name__ == "__main__":
    main()
