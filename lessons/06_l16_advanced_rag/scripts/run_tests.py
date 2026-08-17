"""Тестирование ассистента: 6 запросов разных типов + проверка цитирования.

Запуск: uv run python scripts/run_tests.py [--reset]
Артефакты: results/test_log.md, results/test_results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from pathlib import Path

from corporate_assistant.assistant import build_assistant
from corporate_assistant.config import COLLECTION_NAME, RESULTS_DIR
from corporate_assistant.indexing import qdrant_client
from corporate_assistant.parsing import load_all_chunks, save_chunks
from corporate_assistant.prompts import source_label

CITATION_RE = re.compile(r"\[([^\]]+)\]")


@dataclasses.dataclass
class TestCase:
    name: str
    query: str
    expected_sources: list[str]   # подстроки, которые должны присутствовать в цитатах
    expect_unknown: bool = False  # ожидаем отказ «Я не знаю»


TEST_CASES: list[TestCase] = [
    TestCase(
        name="1. Прямой вопрос (один факт в PDF)",
        query="За сколько дней нужно писать заявление на отпуск?",
        expected_sources=["Инструкция_по_отпускам.pdf"],
    ),
    TestCase(
        name="2. Печатный номер страницы (нумерация сдвинута титулом/оглавлением)",
        query="Какой размер суточных выплачивается при командировке и каков лимит расходов на проживание?",
        expected_sources=["Положение_о_командировках.pdf"],
    ),
    TestCase(
        name="3. Агрегация из двух документов",
        query=(
            "Я уезжаю в командировку, а сразу после неё планирую уйти в отпуск. "
            "Какие сроки подачи заявлений нужно соблюсти для каждого события?"
        ),
        expected_sources=["Положение_о_командировках.pdf", "Инструкция_по_отпускам.pdf"],
    ),
    TestCase(
        name="4. Wiki-статья",
        query="Какие требования к корпоративным паролям и как часто их нужно менять?",
        expected_sources=["Корпоративные пароли и доступы"],
    ),
    TestCase(
        name="5. Out-of-domain (провокация)",
        query="Какая зарплата у Senior Python Developer и какой годовой бонус?",
        expected_sources=[],
        expect_unknown=True,
    ),
    TestCase(
        name="6. Заголовок раздела + вычисление по фактам из одного файла",
        query="Сотруднику со стажем 7 лет открыли больничный. Сколько процентов от заработка он получит?",
        expected_sources=["Инструкция_по_больничным_листам.pdf"],
    ),
]


def check_case(answer: str, case: TestCase, allowed: list[str]) -> dict:
    citations = CITATION_RE.findall(answer)
    missing = [
        src for src in case.expected_sources if not any(src in c for c in citations)
    ]
    invalid = [c for c in citations if c not in allowed]
    if case.expect_unknown:
        unknown_ok = "не знаю" in answer.lower()
        hallucinated = [c for c in citations if "pdf" in c.lower() or "wiki" in c.lower()]
        passed = unknown_ok and not hallucinated and not invalid
        details = (
            f"не-знаю={unknown_ok}, выдуманные_источники={hallucinated or 'нет'}, "
            f"неизвестные_цитаты={invalid or 'нет'}"
        )
    else:
        passed = not missing and not invalid
        details = (
            f"отсутствуют_цитаты={missing or 'нет'}, "
            f"неизвестные_цитаты={invalid or 'нет'}"
        )
    return {"passed": passed, "details": details, "citations": citations, "invalid": invalid}


def build_if_needed(reset: bool) -> None:
    client = qdrant_client()
    if reset or not client.collection_exists(COLLECTION_NAME):
        chunks = load_all_chunks()
        save_chunks(chunks)
        from corporate_assistant.indexing import build_index

        build_index(chunks, reset=reset, collection_name=COLLECTION_NAME)


def format_sources(assistant, query: str) -> str:
    nodes = assistant.retrieve(query)
    lines = []
    for i, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        snippet = " ".join(node.text.split())[:120]
        lines.append(
            f"{i}. {source_label(node.metadata)} | score={node_with_score.score:.3f} | {snippet}"
        )
    return "\n".join(lines)


def run_tests(reset: bool) -> list[dict]:
    build_if_needed(reset)
    assistant = build_assistant(COLLECTION_NAME)
    results: list[dict] = []
    overall_pass = True

    for case in TEST_CASES:
        print(f"\n=== {case.name} ===")
        started = time.monotonic()
        answer = assistant.answer(case.query)
        elapsed = time.monotonic() - started

        allowed = list(
            dict.fromkeys(source_label(s["metadata"]) for s in answer.sources)
        )
        check = check_case(answer.answer, case, allowed)

        print(f"Вопрос: {case.query}")
        print(f"Ответ ({elapsed:.1f} с):\n{answer.answer}")
        if answer.dropped_citations:
            print(f"Отброшено выдуманных ссылок: {answer.dropped_citations}")
        print("Найденные чанки:")
        print(format_sources(assistant, case.query))
        print(f"Вердикт: {'PASS' if check['passed'] else 'FAIL'} — {check['details']}")

        overall_pass &= check["passed"]
        results.append(
            {
                "name": case.name,
                "query": case.query,
                "answer": answer.answer,
                "elapsed_s": round(elapsed, 1),
                "check": check,
                "dropped_citations": answer.dropped_citations,
                "sources": [
                    {
                        "label": source_label(s["metadata"]),
                        "score": round(s["score"], 3) if isinstance(s["score"], float) else None,
                        "text": s["text"],
                    }
                    for s in answer.sources
                ],
                "context": answer.context,
            }
        )

    print(f"\nИТОГ: {'ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if overall_pass else 'ЕСТЬ ПРОВАЛЫ'}")
    return results


def write_report(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Логи тестирования корпоративного ассистента",
        "",
        f"Модель эмбеддингов: `intfloat/multilingual-e5-small` · LLM: `qwen2.5:3b` (Ollama) · Vector DB: Qdrant",
        "",
    ]
    for result in results:
        check = result["check"]
        lines += [
            f"## {result['name']}",
            "",
            f"**Вопрос:** {result['query']}",
            "",
            "**Ответ:**",
            "",
            result["answer"],
            "",
            "**Найденные чанки:**",
            "",
        ]
        for i, source in enumerate(result["sources"], start=1):
            score = source["score"]
            lines.append(f"{i}. **{source['label']}** (score={score})")
            lines.append(f"   `{source['text'][:200]}`")
        lines.append("")
        lines.append(f"**Проверка цитирования:** `{'PASS' if check['passed'] else 'FAIL'}` — {check['details']}")
        if result.get("dropped_citations"):
            lines.append("")
            lines.append(f"**Отброшено выдуманных ссылок:** {', '.join(result['dropped_citations'])}")
        lines.append("")
        lines.append("---")
        lines.append("")

    (RESULTS_DIR / "test_log.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS_DIR / "test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Отчёт сохранён: {RESULTS_DIR / 'test_log.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон тестового набора вопросов.")
    parser.add_argument("--reset", action="store_true", help="Пересобрать индекс перед тестами.")
    args = parser.parse_args()

    results = run_tests(reset=args.reset)
    write_report(results)
    return 0 if all(r["check"]["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
