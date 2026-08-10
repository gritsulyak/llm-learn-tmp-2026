# AI SMM-отдел — мультиагентная система (LangGraph + Ollama, локально, uv)

Мультиагентный пайплайн (Strategist → Copywriter → Editor → Publisher) на **LangGraph**,
работает полностью локально через **Ollama** (CPU, без CUDA), с опциональным трейсингом в **Langfuse**.
Окружение управляется через **uv**.

## Структура проекта

```
05_l13_langfuse_ai_smm/
├── doc
|   ├── TASK_RU.md           # исходное задание
|   └── ARCHITECTURE.md      # модели, структура папок, обоснование
├── src/
│   ├── config.py              # настройки из .env
│   ├── agents/
│   │   ├── strategist/        # prompt.py, node.py, schema.py
│   │   ├── copywriter/
│   │   ├── editor/
│   │   └── publisher/
│   ├── graph/
│   │   ├── state.py             # SMMState (TypedDict)
│   │   ├── build_graph.py        # сборка графа + conditional edges
│   │   └── router.py              # route_after_editor, route_after_publisher
│   ├── llm/
│   │   ├── ollama_client.py        # ChatOllama клиент
│   │   └── models_config.py         # маппинг роль -> модель
│   └── observability/
│       └── langfuse_setup.py         # CallbackHandler для Langfuse
├── scripts/run_pipeline.py             # точка входа CLI
├── tests/                                # pytest, unit-тесты с моками LLM
├── pyproject.toml                          # зависимости и настройки (uv)
├── uv.lock                                   # зафиксированные версии
└── .env.example
```

## 1. Установка uv (если ещё не установлен)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# перезапустите шелл или: source $HOME/.local/bin/env
```

## 2. Установка Ollama и моделей (CPU, слабое железо)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Модели по ролям (см. docs/ARCHITECTURE.md — выбраны под CPU/AMD GPU)
ollama pull qwen2.5:3b
ollama pull lakomoor/vikhr-llama-3.2-1b-instruct:1b
ollama pull qwen2.5:1.5b
```

Если `vikhr-llama-**` не находится в реестре — импортируйте GGUF через `ollama create`,
либо замените в `.env` на `qwen2.5:3b` для всех ролей.

## 3. Настройка окружения (uv)

```bash
cd 05_l13_langfuse_ai_smm

# создаёт .venv и ставит зависимости из pyproject.toml / uv.lock
uv sync --group dev
```

`uv` сам создаёт и активирует виртуальное окружение через `uv run` — отдельно
активировать `.venv` не обязательно, но можно:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 4. Конфигурация

```bash
cp .env.example .env
```

Отредактируйте `.env` при необходимости (модели, нишу, соцсеть, лимит итераций).
Langfuse-ключи можно оставить пустыми — трейсинг тогда просто отключается.

## 5. Запуск пайплайна

```bash
# базовый запуск (ниша и соцсеть берутся из .env)
uv run python -m scripts.run_pipeline

# с явными параметрами
uv run python -m scripts.run_pipeline \
  --niche "Онлайн-школа английского языка" \
  --social-network VK \
  --max-revision-rounds 3

# только вывести граф в формате mermaid (без вызова LLM)
uv run python -m scripts.run_pipeline --graph
```

Результат сохраняется в `data/logs/run_<timestamp>.json`: полный `content_plan`,
переписка (draft/comments по каждой итерации) и финальные посты от Publisher.

## 6. Наблюдаемость (Langfuse, опционально)

### 6.1 Запуск Langfuse через Docker

Клонируйте официальный репозиторий Langfuse и поднимите `docker-compose`:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

> Примечание: если не тянется образ `cgr.dev/chainguard/minio` (ошибка `403 Forbidden`),
> замените в `docker-compose.yml` строку
> `image: cgr.dev/chainguard/minio` на `image: minio/minio:latest`
> (официальный образ MinIO также содержит `mc`/`curl`, healthcheck менять не нужно).

После старта веб-интерфейс доступен на `http://localhost:3000`.

### 6.2 Получение ключей API

1. Откройте `http://localhost:3000` и зарегистрируйте первый аккаунт — он автоматически
   становится владельцем и создаёт проект по умолчанию.
2. Перейдите **Settings → API Keys** (внутри проекта) и нажмите **New API Key**.
3. Скопируйте выданные значения:
   - **Public Key** — `pk-...`
   - **Secret Key** — `sk-...` (показывается только один раз при создании, сохраните сразу)

### 6.3 Настройка `.env`

Впишите ключи в `.env` проекта:

```
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=http://localhost:3000
```

Если ключи пустые — трейсинг автоматически отключается (`CallbackHandler` не создаётся).
При заданных ключах `CallbackHandler` подключается к графу LangGraph, и в Langfuse
появляется полный Trace с шагами каждого агента, задержками и токенами.

### 6.4 Проверка

Запустите пайплайн и откройте раздел **Traces** в веб-интерфейсе Langfuse
(`http://localhost:3000/traces`) — должен появиться свежий Trace с шагами
Strategist → Copywriter → Editor → Publisher.

## 7. Тесты

```bash
uv run pytest -q
```

Тесты используют моки LLM (`unittest.mock.patch`) и не требуют запущенного Ollama —
проверяются граф, роутинг условных переходов и парсинг JSON-ответов агентов.
Все 11 тестов проходят без обращения к сети/Ollama.

## Модели по ролям (напоминание)

| Роль | Модель | Причина |
|------|--------|---------|
| Strategist | `qwen2.5:3b` | лучше держит структуру/JSON |
| Copywriter | `vikhr-llama-3.2-3b` | сильнее в креативном русском тексте |
| Editor | `qwen2.5:3b` | стабильнее следует инструкциям критики |
| Publisher | `qwen2.5:1.5b` | задача формальная, минимума достаточно |

Подробности — в `docs/ARCHITECTURE.md`.
