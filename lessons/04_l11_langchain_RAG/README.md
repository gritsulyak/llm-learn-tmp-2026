# Enterprise Private GPT — локальный RAG на LangChain + Ollama + Qdrant
Защищённый корпоративный чат-бот (RAG), работающий полностью локально:
LLM через **Ollama**, векторная база **Qdrant** (docker-compose), эмбеддинги
**HuggingFace sentence-transformers** (CPU). Ни один запрос не уходит в интернет —
подходит для конфиденциальных корпоративных документов.

## Структура проекта

```
enterprise_rag_langchain/
├── README.md                  # этот файл
├── requirements.txt           # зависимости (venv)
├── .env.example                # шаблон конфигурации (скопировать в .env)
├── docker-compose.yml          # Qdrant (векторная БД)
├── doc/                       # техническая документация
│   ├── ARCHITECTURE.md
│   ├── RUN_AND_TEST.md
│   └── NOTEBOOK_CONVERSION.md
├── notebook/                   # Jupyter-лаборатория (research/эксперименты)
│   └── rag_lab.ipynb
├── src/                        # исходный код приложения
│   ├── config.py
│   ├── ingest.py                # загрузка PDF/TXT -> чанки -> Qdrant
│   ├── rag_chain.py              # RAG-цепочка (LCEL): retriever + Ollama LLM
│   └── app.py                    # Gradio-интерфейс
├── scripts/
│   └── check_ollama.sh          # проверка Ollama без интернета
├── tests/
│   └── test_scenarios.py        # 3 тестовых сценария из TASK_RU.md
├── docs_corpus/ (создайте сами) # ваши PDF/TXT книги-документы для базы знаний
└── data/qdrant_storage/          # том с данными Qdrant (создаётся автоматически)
```

> Папка с вашими PDF-книгами (`docs/` по умолчанию в `.env`) — положите туда файлы
> перед запуском `ingest.py`. Можно переименовать путь через `DOCS_DIR` в `.env`.

## Выбор модели Ollama

Из вашего списка `ollama ls` для этой задачи (CPU-only ноутбук, RAG с чанками
до ~800 токенов, русский+английский язык) рекомендуется:

| Модель | Размер | Комментарий |
|---|---|---|
| qwen3.5:4b  | 3.4 GB | Компактная, быстрая на CPU, хорошо работает с RU/EN, современная архитектура — оптимальный баланс скорости и качества для RAG на ноутбуке без GPU |
| yandex/YandexGPT-5-Lite-8B | 4.9 GB | Хорош для русского языка, но 8B тяжелее и медленнее на CPU |
| **llama3:latest** | 4.7 GB | Общего назначения, слабее на русском по сравнению с YandexGPT/Qwen |

Лучше всего по скорости и для небольших запросов показала себя llama3, ее и используем.

Модель эмбеддингов — `paraphrase-multilingual-MiniLM-L12-v2`
(мультиязычная, ~470 MB, быстрая на CPU, поддерживает RU+EN документы).

## Быстрый старт

```bash
# 1. Создать и активировать venv
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Скопировать конфиг
cp .env.example .env

# 4. Скачать модель в Ollama (если ещё нет)
ollama pull qwen2.5:1.5b
ollama create qwen2.5-32k -f ./Modelfile
ollama run qwen2.5-32k

# 5. Поднять Qdrant
docker compose up -d

# 6. Положить PDF/TXT документы в папку docs/
mkdir -p docs && cp /path/to/your/*.pdf docs/

# 7.1 Почистить quadrant  
curl -X DELETE http://localhost:6333/collections/enterprise_docs

# 7.2 Прогнать индексацию документов
python -m src.ingest

# 8. Запустить чат-интерфейс (Gradio)
python -m src.app
# откроется http://127.0.0.1:7860

# 9. (опционально) прогнать тестовые сценарии из ДЗ
python -m tests.test_scenarios
```

Подробности запуска и тестирования — в `doc/RUN_AND_TEST.md`.
Описание архитектуры — в `doc/ARCHITECTURE.md`.
Конвертация notebook <-> script — в `doc/NOTEBOOK_CONVERSION.md`.

## Почему это защищает корпоративные данные

Все компоненты (LLM через Ollama, эмбеддинги через HuggingFace локально,
векторная БД Qdrant в docker-compose) работают на localhost без обращения
к внешним API. Единственный риск утечки — если сам ноутбук скомпрометирован
или Qdrant/Ollama случайно открыты наружу (`0.0.0.0`) без файрвола; в докер-компоузе
порты проброшены только на `localhost` по умолчанию.

