"""
Тестовые сценарии из TASK_RU.md, пункт 4 "Анализ слабых мест".

Запуск:
    python -m tests.test_scenarios
"""
import json
from pathlib import Path
import sys

from src.rag_chain import RagBot
from src.ingest import run_ingest
from src.config import TOP_K, QDRANT_TEST_COLLECTION, TEST_DOCS_DIR

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag_chain import RagBot
from src.config import TOP_K

# 1. Запрос по базе (замените на реальный вопрос по вашим документам)
Q_IN_BASE = "Какой регламент действий при инциденте информационной безопасности?"

# 2. Запрос вне базы
Q_OUT_OF_BASE = "Дай рецепт классического яблочного пирога."

# 3. Конфликт фактов (LLM может знать общий ответ, но в базе - другой)
Q_CONFLICT = "Сколько дней отпуска положено сотруднику по стандартным нормам трудового права?"


def run_case(bot: RagBot, label: str, question: str):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"Вопрос: {question}")
    result = bot.ask(question)
    print(f"Ответ: {result['answer']}")
    print(f"Источники: {result['sources']}")
    print(f"Latency: {result['latency_sec']} сек | чанков: {result['num_chunks']}")
    return {"label": label, "question": question, **result}


def main():
    print(f"Индексация тестового корпуса '{TEST_DOCS_DIR}' -> коллекция '{QDRANT_TEST_COLLECTION}'...")
    run_ingest(TEST_DOCS_DIR, QDRANT_TEST_COLLECTION)

    bot = RagBot(top_k=TOP_K, collection_name=QDRANT_TEST_COLLECTION)
    
    results = []
    results.append(run_case(bot, "1. Запрос ПО базе", Q_IN_BASE))
    results.append(run_case(bot, "2. Запрос ВНЕ базы", Q_OUT_OF_BASE))
    results.append(run_case(bot, "3. Конфликт фактов", Q_CONFLICT))

    # Эксперимент с Top-K
    print(f"\n{'='*70}\nЭксперимент: влияние Top-K на latency\n{'='*70}")
    for k in (1, 3, 6):
        bot.set_top_k(k)
        r = bot.ask(Q_IN_BASE)
        print(f"Top-K={k}: latency={r['latency_sec']} сек, чанков={r['num_chunks']}")
        results.append({"label": f"topk_experiment_k{k}", **r})

    out_path = Path(__file__).resolve().parent / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Результаты сохранены в {out_path}")


if __name__ == "__main__":
    main()
