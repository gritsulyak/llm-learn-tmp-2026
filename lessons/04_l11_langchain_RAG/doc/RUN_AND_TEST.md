# Запуск и тестирование

## 1. Развертывание локальной LLM (Ollama)

Если нужна именно ollama вместо llama3 (для которой аналогично)
```bash
ollama pull qwen3.5:4b
bash scripts/check_ollama.sh qwen3.5:4b
```

Скрипт проверяет: список моделей, ответ модели в консоли, доступность
API `http://localhost:11434` — без обращения во внешний интернет.

### Вариант через Yandex Cloud

Если вместо Ollama используете Yandex Cloud (YandexGPT API или
развернутую в облаке модель), замените в `.env`:

```
OLLAMA_BASE_URL=<адрес вашего Yandex Cloud эндпоинта>
OLLAMA_LLM_MODEL=<имя модели в облаке>
```

и убедитесь, что `src/rag_chain.py::create_llm` использует совместимый
клиент (для Yandex Cloud Foundation Models потребуется заменить
`ChatOllama` на `langchain_community.chat_models.ChatYandexGPT` —
интерфейс LCEL остаётся тем же).

## 2. Векторная база и обработка документов

```bash
docker compose up -d          # поднимает Qdrant на localhost:6333
mkdir -p docs
cp /path/to/pdf_books/*.pdf docs/
python -m src.ingest
```

`ingest.py` рекурсивно ищет `*.pdf` и `*.txt` в `docs/`, режет их
`RecursiveCharacterTextSplitter` (chunk_size=800, overlap=120), считает
эмбеддинги (`paraphrase-multilingual-MiniLM-L12-v2`, CPU) и пишет в
коллекцию Qdrant `enterprise_docs` вместе с метаданными `source_file`.

## 3. RAG-пайплайн

```bash
python -m src.rag_chain
```

Запускает консольный чат: вопрос -> векторизация -> semantic search (Top-K
из `.env`) -> подстановка контекста в системный промпт -> генерация ответа
локальной LLM. Выводит также источники и latency.

## 4. Интерфейс и тестирование слабых мест

```bash
python -m src.app
```

Откройте `http://127.0.0.1:7860`. В интерфейсе есть слайдер Top-K для
экспериментов на лету.

Прогон трёх обязательных тестовых сценариев из ДЗ (п.4):

```bash
python -m tests.test_scenarios
```

Скрипт задаёт по очереди:

1. **Вопрос строго по базе** — ответ должен содержать факты из документов
   и ссылку на source_file.
2. **Вопрос вне базы** (рецепт пирога) — бот должен явно отказаться отвечать.
3. **Конфликт фактов** — сверяет, что бот берёт данные из корпоративной базы,
   а не из общих знаний модели.

Плюс эксперимент с Top-K = 1, 3, 6 — фиксирует latency и число чанков в
`tests/test_results.json` для отчёта.

## 5. в отчёт (README/notebook)

- Скриншот/скринкаст интерфейса Gradio с реальными вопросами.
- Три примера ответов (по базе / вне базы / конфликт) из
  `tests/test_results.json`.
- Вывод о выбранных моделях (см. README.md, раздел "Выбор модели Ollama")
  и трудностях локального развертывания (см. `docs/ARCHITECTURE.md`,
  раздел "Ограничения").
