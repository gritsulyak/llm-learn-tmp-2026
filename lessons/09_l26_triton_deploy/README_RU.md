# Triton Inference Server: оркестрация моделей (домашнее задание)

Задание: [TASK.md](./TASK.md). Отчёт по результатам: [REPORT.md](./REPORT.md).
Английская версия: [README.md](./README.md).

## Контекст

Домашнее задание по оркестрации моделей в NVIDIA **Triton Inference Server**.
Нужно развернуть пайплайн из трёх разнородных лёгких моделей на разных бэкендах,
объединить их Business Logic Scripting (BLS) моделью, запустить в Docker,
провести нагрузочное тестирование и собрать отчёт.

## Архитектура

| Модель | Бэкенд | Модель | Параметры | Роль |
|--------|---------|--------------------|--------|------|
| `text_tokenizer` | python | `huawei-noah/TinyBERT_General_4L_312D` tokenizer | — | Токенизация |
| `classification` | onnxruntime | TinyBERT 4L-312D (ONNX) | 14.5M | Классификация вопрос/утверждение |
| `embedding` | onnxruntime | `sentence-transformers/all-MiniLM-L6-v2` (ONNX) | 22.7M | Эмбеддинги 384-d |
| `generation` | python (HTTP) | llama.cpp (`llama-server`) + Qwen2.5-0.5B-Instruct-Q4_K_M.gguf | 0.5B | Генерация ответа на **Vulkan iGPU** |
| `ensemble` | python (BLS) | — | — | Оркестрация цепочки |

```
Входной текст
   │
   ▼
text_tokenizer ──► embedding (mean-pool + L2-norm)
   │
   └────────────► classification
                     │
              [утверждение] ─► ответ "[Classification] ..."
              [вопрос]      ─► generation ─► ответ "[Generated] ..."
```

BLS-модель (`ensemble`) вызывает остальные модели внутри Triton через
`pb_utils.InferenceRequest` / `.exec()`, поэтому клиент делает **один** запрос
и получает финальный результат пайплайна.

## Предварительные требования

- Docker с Compose v2 (`docker compose version`)
- Доступ к `nvcr.io/nvidia/tritonserver` — **нужен бесплатный вход в NGC**
  для pull образа (шаг 3 ниже)
- Python 3.10+ и [uv](https://docs.astral.sh/uv/)
- Рекомендуется от 8 ГБ RAM и 4+ ядер CPU
- Работает без NVIDIA GPU. Генерация ускоряется на **AMD iGPU** через
  **llama.cpp + Vulkan** (`/dev/dri`, Radeon 780M).
- Для генерации нужен запущенный на хосте **llama.cpp `llama-server`**
  (Vulkan-сборка) — Triton-модель `generation` шлёт ему HTTP
  (`LLAMA_SERVER_URL`, по умолчанию `http://host.docker.internal:8080`).
  Без него генерация откатывается на CPU-`transformers`.

## Настройка

### 1. Установка Python-зависимостей (uv)

```bash
uv sync
```

Устанавливает зависимости из lockfile с CPU-only PyTorch (`torch` берётся из
индекса `pytorch-cpu` — без CUDA-колёс).

### 2. Экспорт ONNX-моделей

```bash
uv run python export_models.py
```

Скачивает TinyBERT и MiniLM, экспортирует обе в ONNX и сохраняет в
`model_repository/{classification,embedding}/1/`. Модели `text_tokenizer` и
`ensemble` работают на Python-бэкенде и экспорт не требуют. Модель
`generation` обслуживает **GGUF** через llama.cpp: положите `model.gguf`
(например, `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, ~0.4 ГБ) в
`model_repository/generation/1/model.gguf`.

> **Важно про модель классификации.** TinyBERT — *generic pretrained*
> чекпоинт: его head классификатора инициализирован случайно и без обучения
> возвращает почти нулевые logits (confidence ~0.00x — это шум). Поэтому
> классификатор в этом домашнем задании был **дообучен на маленьком датасете
> из ~290 утверждений/вопросов** (3 эпохи, CPU). Перезапустить в любой момент:
>
> ```bash
> uv run python train_classifier.py     # дообучение + экспорт ONNX в model_repository/classification/1/
> docker compose restart                # перезагрузка модели в Triton
> ```
>
> После обучения мы получаем осмысленные вероятности (например,
> `confidence=0.707`), а не «монетку». Сырой вывод `export_models.py`
> для классификации использовать нельзя.

### 3. Pull Triton-образа (NGC)

Образ `nvcr.io/nvidia/tritonserver:24.01-py3` лежит в реестре NVIDIA NGC,
который блокирует анонимный pull (**403 Forbidden**). Бесплатного аккаунта
достаточно:

1. Регистрация: <https://ngc.nvidia.com/signup> (бесплатно).
2. Сгенерировать API-ключ: <https://ngc.nvidia.com/setup/api-key>.
3. Войти — имя пользователя всегда `$oauthtoken`, пароль — API-ключ:

   ```bash
   docker login nvcr.io
   # Username: $oauthtoken
   # Password: <NGC API ключ>
   ```

4. Проверить, что pull работает, до запуска compose:

   ```bash
   docker pull nvcr.io/nvidia/tritonserver:24.01-py3
   ```

   Если у вас доступен *другой* Triton-образ, переопределите тег без правки
   репозитория: `TRITON_IMAGE=<ваш-образ> docker compose up -d`. То же самое
   для SDK-образа Perf Analyzer: `SDK_IMAGE=<...> ./perf/run_perf_analyzer.sh`.

## Запуск

### Запуск llama.cpp `llama-server` (Vulkan на AMD iGPU) — один раз

Модель `generation` отвечает по HTTP из локального llama.cpp. Запустите его на
хосте (бинарник должен быть **Vulkan**-сборкой):

```bash
bash perf/start_llama_server.sh
# curl http://127.0.0.1:8080/health   → {"status":"ok"}
```

По умолчанию: модель `model_repository/generation/1/model.gguf`, `-ngl 99`
(весь слой-стек на iGPU), порт `8080`, `-np 4` параллельных слотов.
Переопределяется env: `PORT=8090 MODEL=/путь/к/model.gguf`.

### Запуск Triton (Docker)

```bash
docker compose build      # ставит CPU-torch, transformers, numpy==1.26.4
docker compose up -d
```

- монтирует `./model_repository` в `/models`
- пробрасывает порты **8000** (HTTP), **8001** (gRPC), **8002** (метрики)
- запускает `tritonserver --model-repository=/models`
- `extra_hosts` резолвит `host.docker.internal` → хост, чтобы `generation`
  достучался до `llama-server` (поддержан override `LLAMA_SERVER_URL`)

### Остановка

```bash
docker compose down
```

## Порядок ручной проверки (выполняется по шагам)

1. **Health check** — сервер запущен:
   ```bash
   curl -vvv http://localhost:8000/v2/health/live
   curl -vvv http://localhost:8000/v2/health/ready
   ```
   Обе команды вернут HTTP 200, когда сервер готов (Triton 24.01 возвращает
   пустое тело, а не `1`).

2. **Все 5 моделей в статусе READY** — дождаться статуса `READY`
   (первый prompt-eval в llama.cpp прогревает шейдеры/GPU-состояние):
   ```bash
   curl -vvv -X POST http://localhost:8000/v2/repository/index
   ```
   Ожидается: `text_tokenizer`, `classification`, `embedding`, `generation`,
   `ensemble` — все `state: "READY"`. (В Triton 24.01 `GET /v2/models`
   возвращает 404.)

3. **Smoke-тест всей цепочки одним запросом**:
   ```bash
 
   # → "[Generated] ..."   (вопрос уходит в генерацию)
   uv run python client.py "I love this product."
   # → "[Classification] label=..., confidence=..."
   ```

4. **Прогнать тесты** (см. ниже).

5. **Нагрузочное тестирование** (см. ниже) — Perf Analyzer и/или `perf/load_test.py`.

6. **Проверить метрики Triton на порту 8002**:
   ```bash
   curl http://localhost:8002/metrics | head
   uv run python perf/metrics.py
   ```

## Тесты

### 1. Unit-тесты (офлайн — без Triton-сервера)

Проверяют логику маршрутизации BLS на mock-модуле `pb_utils`:

```bash
uv run pytest tests/ -v
```

Покрытие: маршрут утверждение→классификатор, вопрос→генерация, вызовы
эмбеддинга/токенизатора, передача промпта в генерацию, обработка батча,
ошибка при отсутствии зависимой модели.

### 2. E2E-тесты (требуют запущенного Triton-сервера)

```bash
# против уже запущенного сервера
uv run pytest tests/test_e2e.py -v

# или пусть pytest сам поднимет docker-стек (и погасит его после)
TRITON_START_DOCKER=1 uv run pytest tests/test_e2e.py -v
```

Покрытие: сервер жив, все 5 моделей в READY, один запрос к `ensemble` даёт
валидный ответ, прямой инференс `generation` и цепочек
токенизатор→классификатор / токенизатор→эмбеддинг. Тесты **корректно
пропускаются**, если сервер недоступен.

### 3. Интеграционный скрипт (tritonclient, против живого сервера)

```bash
uv run python test_ensemble.py
```

Проверка каждой модели и полного BLS-пайплайна через gRPC.

## Нагрузочное тестирование (часть 4)

### NVIDIA Perf Analyzer (SDK-образ)

```bash
./perf/run_perf_analyzer.sh
```

Запускает `perf_analyzer` из Triton SDK-контейнера против
`text_tokenizer`, `embedding`, `generation`, `ensemble`
(concurrency 1:4, p95, окна по 5 c). Для `classification` нужен
заранее токенизированный вход — вместо этого используйте Python-тестер ниже.

### Python-тестер нагрузки (без SDK)

```bash
uv run python perf/load_test.py --model ensemble --concurrency 4 --duration 20
uv run python perf/load_test.py --all --concurrency 2 --duration 10 --out perf/results/results.json
```

Выводит RPS и среднее/p50/p95/p99 время ответа по каждой модели. Модели
с токенизированным входом (`classification`, `embedding`) кормятся через
локальный токенизатор.

### Разбор задержек (метрики Triton)

```bash
uv run python perf/metrics.py
```

Показывает по каждой модели число инференсов, суммы задержек Queue и
Compute Infer из Prometheus-эндпоинта на порту 8002 — помогает найти узкие места.

## Отчёт (часть 5)

См. `REPORT.md` — дерево репозитория, сводка конфигураций, методология,
таблицы результатов (заполните своими замерами), анализ узких мест и выводы
(Triton/BLS против Flask/FastAPI, варианты оптимизации).

## Структура репозитория

```
├── client.py                  # BLS-клиент (один запрос)
├── docker-compose.yaml        # сервер Triton (порты 8000/8001/8002)
├── Dockerfile                 # образ с transformers/optimum
├── export_models.py           # экспорт ONNX для classification/embedding
├── train_classifier.py        # дообучение head TinyBERT + экспорт ONNX
├── model_repository/          # все модели: config.pbtxt + папки версий
├── perf/                      # инструменты нагрузочного тестирования
├── perf/start_llama_server.sh # запуск llama.cpp llama-server (Vulkan iGPU)
├── perf/results/              # (результаты прогонов, в gitignore)
├── pyproject.toml             # uv-проект / зависимости
├── reference/                 # референсное решение задания
├── REPORT.md                  # шаблон отчёта по ДЗ
├── test_ensemble.py           # интеграционные проверки по gRPC
├── tests/                     # pytest: unit (офлайн) + e2e (живой сервер)
└── README_RU.md / README.md
```

## Примечания / решение проблем

- **Pull образа отклонён (403)** для `nvcr.io` — пройдите NGC-логин из шага 3
  раздела «Pull Triton-образа», либо используйте доступный вам Triton-образ:
  `TRITON_IMAGE=<ваш-образ> docker compose up -d` (аналогично для SDK-образа
  Perf Analyzer через `SDK_IMAGE`).
- **Модель зависла в NOT READY** — смотрите `docker compose logs -f triton`;
  проверьте, что запущен `bash perf/start_llama_server.sh` (иначе генерация
  откатится на медленный CPU-`transformers`, ~23 с за ответ).
- **Triton 24.01 + `numpy 2.x`** — Python backend отдаёт пустые выходные
  тензоры (0 raw-байт), пока в образе не зафиксирован `numpy==1.26.4`
  (в Dockerfile уже зафиксирован).
- **Строковые (`TYPE_STRING`) входы не работают** в Python backend Triton 24.01
  (issue #7391). Поэтому текст передаётся как UTF-8-байты в `UINT8`-тензоре
  (см. `client.py`/`tests`); не возвращайте `BYTES` без апгрейда Triton.
- По умолчанию **CPU-only**; для NVIDIA GPU поставьте `kind: KIND_GPU` в каждом
  `config.pbtxt` и уберите лимиты CPU/памяти из compose.
- Размерности выходов **динамические** (`dims: [-1]` / `[-1, 128]`), паддинг
  до `max_length=128`; в конфигах и клиентах нет жёстких фиксированных форм.

## Лицензия / рамки

Учебный проект для практики с Triton Inference Server. NVIDIA GPU *не*
требуется — весь стек работает на CPU (совместимо с AMD iGPU).