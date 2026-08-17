# Corporate Knowledge Assistant — ДЗ «Продвинутый RAG»

Корпоративный ассистент по внутренней базе документов АО «Осинка»: ищет ответы
в PDF-инструкциях и Wiki-статьях, отвечает только по контексту и подтверждает
каждый факт ссылкой на источник вида `[Файл.pdf, стр. X]` / `[Название Wiki-статьи]`.

Базовый репозиторий: [Ilia2704/rag-llamaindex](https://github.com/Ilia2704/rag-llamaindex).

## Стек

| Компонент      | Решение                                                               |
|----------------|-----------------------------------------------------------------------|
| Парсинг PDF    | PyMuPDF — текст по страницам + колонтитулы (заголовок + печатный номер)|
| Парсинг Wiki   | BeautifulSoup                                                        |
| Чанкинг        | Сплиттер по границам предложений; чанк привязан к одной странице      |
| Эмбеддинги     | `intfloat/multilingual-e5-small` (384 d, CPU)                         |
| Vector DB      | Qdrant `corporate_assistant_2026`                                     |
| LLM            | `yandexgpt-lite` через Yandex Cloud (ретривер + генерация)         |
| Фреймворк      | LlamaIndex                                                           |
| Оценка RAGAS   | `ragas` + судья: Yandex Cloud (`yandexgpt-lite`) или Ollama (`qwen2.5:3b`) |

## Структура проекта

```
src/corporate_assistant/
  config.py        Константы (URL, модели, размеры чанков)
  parsing.py       PyMuPDF + bs4: извлечение текста и метаданных, чанкинг
  indexing.py      Эмбеддинги, сборка/чтение Qdrant-индекса, retriever
  prompts.py       Системный промпт, форматирование контекста, grounding
  assistant.py     RAG-пайплайн (retrieval → LLM → проверка ссылок)
  llm.py           Обёртка над Yandex Cloud (CustomLLM)
scripts/
  generate_knowledge_base.py  Генерация PDF и HTML wiki (reportlab)
  build_index.py              Парсинг + индексация в Qdrant
  run_tests.py                Прогон 6 тестовых вопросов с проверкой цитирования
  experiment_chunk_size.py    Эксперимент: влияние chunk_size на точность страниц
  chat.py                     Интерактивный чат с источниками
  evaluate_ragas.py           Оценка RAGAS: faithfulness, context_precision, context_recall
tests/
  test_core.py     Pytest: парсер, чанкинг, форматирование, grounding
notebooks/
  report.py        Полный отчёт — запускается как скрипт или ячейками (VS Code %%)
data/
  raw/pdfs/        PDF-инструкции (4 шт., генерируются скриптом)
  raw/wiki/        Wiki-страницы в HTML (4 шт.)
  chunks/          Дамп чанков с метаданными (chunks.json)
results/
  test_log.md      Лог 6 тестовых вопросов с вердиктами
  test_results.json Результаты в машиночитаемом виде
  ragas_scores_ollama.json  Оценка RAGAS — судья Ollama
  ragas_scores_yandex.json  Оценка RAGAS — судья Yandex Cloud
```

## Быстрый старт

```bash
docker compose up -d qdrant       # Qdrant на :6333
ollama pull qwen2.5:3b
uv sync
uv run python scripts/generate_knowledge_base.py   # опционально
uv run python scripts/build_index.py --reset       # парсинг + индексация
uv run python scripts/run_tests.py                 # 6 тестов с проверкой цитирования
uv run pytest -q                                   # unit-тесты парсера и grounding
uv run python scripts/evaluate_ragas.py --judge yandex  # RAGAS-оценка (Yandex судья)
uv run python scripts/evaluate_ragas.py --judge ollama   # RAGAS-оценка (Ollama судья)
```

## Чат-режим

```bash
uv run python scripts/chat.py
# опции: --collection corporate_assistant_2026 (по умолчанию), --top-k 5
# выход: пустая строка, exit, quit или Ctrl+D
```

Каждый ответ сопровождается списком использованных источников с указанием
раздела и страницы: PDF — `Имя_файла.pdf — раздел «Заголовок раздела», стр. N`,
Wiki — `Название статьи — раздел «Заголовок раздела»`. Пример диалога —
в отчёте (`notebooks/report.py`, реальный прогон).

## RAGAS-оценка

Метрики `faithfulness`, `context_precision`, `context_recall` на 5 вопросах с
эталонными ответами. Судья — Yandex Cloud (`yandexgpt-lite`) или локальный
Ollama (`qwen2.5:3b`). Эмбеддинги — локальные (Ollama).
Цитаты `[Файл.pdf, стр. N]` из ответов перед оценкой удаляются (ссылка —
не факт из контекста, судья трактует её как «неподтверждённое утверждение»).

```bash
uv run python scripts/evaluate_ragas.py --judge yandex                          # yandexgpt-lite
uv run python scripts/evaluate_ragas.py --judge yandex --judge-model yandexgpt-5.1
uv run python scripts/evaluate_ragas.py --judge ollama                           # qwen2.5:3b
```

### Сравнение судей (прогон от 2026-08-17)

Один и тот же RAG-пайплайн (Yandex `yandexgpt-lite` как основная LLM),
разные модели-судьи для оценки:

| Метрика            | Ollama (`qwen2.5:3b`) | Yandex (`yandexgpt-lite`) | Yandex (`yandexgpt-5.1`) |
|--------------------|------------------------|----------------------------|---------------------------|
| `faithfulness`     | 0.567                  | 0.900                      | **1.000**                 |
| `context_precision`| 0.951                  | 0.846                      | 0.567                     |
| `context_recall`   | 1.000                  | 1.000                      | 1.000                     |

- `context_recall` = 1.0 для всех судей — ретривер находит всю информацию
- `faithfulness`: `yandexgpt-5.1` ставит **1.0** (все ответы полностью
  опираются на контекст), `yandexgpt-lite` — **0.90** (немного занижает),
  `qwen2.5:3b` — **0.57** (существенно занижает, трактуя ссылки-цитаты
  как галлюцинации)
- `context_precision`: `qwen2.5:3b` — **0.95** (завышает, менее требователен),
  `yandexgpt-lite` — **0.85** (умеренно строгий), `yandexgpt-5.1` — **0.57**
  (очень строгий: считает, что значительная часть retrieved-чанков содержит
  смежную, но не целевую информацию). Разница между `yandexgpt-lite` (0.85)
  и `yandexgpt-5.1` (0.57) в ~28 п.п. указывает на то, что более сильная
  модель-судья предъявляет повышенные требования к точности ретривера:
  засчитываются только чанки, содержащие непосредственно нужный факт
- **Вывод:** сила модели-судьи критически влияет на оценку. `qwen2.5:3b`
  ненадёжна как судья (завышает precision, занижает faithfulness).
  `yandexgpt-5.1` — самый строгий судья: подтверждает идеальную faithfulness,
  но выявляет проблемы с precision ретривера. Для продакшена рекомендуется
  оценка сильной моделью (YandexGPT-5.1 / GPT-класса)

## Отчёт

Полный отчёт с анализом парсинга, цитирования и борьбы с галлюцинациями —
в [`notebooks/report.py`](notebooks/report.py). Запуск:

```bash
uv run python notebooks/report.py        # целиком
# или ячейками в VS Code (каждый блок # %% — отдельная ячейка)

# Сформировать исполненный .ipynb с результатами (ноутбук с выводами всех ячеек):
uv run python scripts/run_notebook.py --report notebooks/report.py \
    --output notebooks/report_run_local_20260817.ipynb
```

## Критерии сдачи

- [x] Парсер извлекает метаданные из PDF (`page`/`header`) и Wiki (`source`/`header`)
- [x] Ответ строго по контексту; out-of-domain → «Я не знаю»
- [x] К каждому факту прикреплена корректная ссылка, проверенная через grounding
