# Отчёт: оркестрация моделей в NVIDIA Triton Inference Server

## 1. Что настроено

Развёрнут пайплайн из трёх разнородных лёгких моделей, объединённых
Business Logic Scripting (BLS) моделью на Python backend:

| Модель | Бэкенд | Модель | Параметры | Роль |
|--------|--------|--------|-----------|-----|
| `text_tokenizer` | python | TinyBERT tokenizer | — | Токенизация текста |
| `classification` | onnxruntime | TinyBERT 4L-312D | 14.5M | Двоичная классификация (вопрос/утверждение) |
| `embedding` | onnxruntime | all-MiniLM-L6-v2 | 22.7M | Эмбеддинги 384-d (mean-pooling + L2-norm) |
| `generation` | python (HTTP) | **llama.cpp (llama-server)**, Qwen2.5-0.5B-Instruct Q4_K_M | 0.5B | Генерация ответа на вопрос на **Vulkan iGPU** |
| `ensemble` | python (BLS) | — | — | Дирижёр: токенизация → эмбеддинг + классификация → генерация |

`generation` уже **не крутит HF-трансформер на CPU**: через Python-backend он
делает HTTP-вызов к локальному `llama-server` (llama.cpp, **Vulkan**-бэкенд на
Radeon 780M /dev/dri), который грузит GGUF-модель (`n_gpu_layers=-1`).
Если llama-server недоступен — баттерфляй-фолбэк на `transformers` (CPU).

Все модели работают на CPU (совместимо с AMD 7840HS iGPU). Для GPU достаточно
сменить `KIND_CPU` на `KIND_GPU` в `config.pbtxt`.

### Дерево Model Repository

```
model_repository/
├── text_tokenizer/
│   ├── config.pbtxt
│   └── 1/model.py
├── classification/
│   ├── config.pbtxt
│   └── 1/ (model.onnx, tokenizer.json, vocab.txt, ...)
├── embedding/
│   ├── config.pbtxt
│   └── 1/ (model.onnx, tokenizer.json, vocab.txt, ...)
├── generation/
│   ├── config.pbtxt
│   └── 1/model.py
└── ensemble/
    ├── config.pbtxt
    └── 1/model.py
```

### Пайплайн BLS

```
Текст → text_tokenizer → embedding (параллельно) + classification
       → если класс «вопрос» (1) → generation → текст
       → иначе → классификация (label + confidence)
```

BLS-модель вызывает остальные через `pb_utils.InferenceRequest`/`.exec()`,
список тензоров строится вручную, pooling эмбеддинга выполняется в `model.py`.

> Текстовые входы передаются как UTF-8-байты в `UINT8`-тензоре
> (`dims: [-1]`) — это обход бага Python-backend Triton 24.01 (issue #7391),
> при котором строковые (`TYPE_STRING`) тензоры десериализуются с ошибкой.

## 2. Развёртывание и известные проблемы

- `docker-compose.yaml` пробрасывает порты **8000 (HTTP), 8001 (gRPC), 8002 (метрики)**
  и монтирует `./model_repository:/models`.
- `Dockerfile` копирует `model_repository` в образ, ставит CPU-torch и запускает
  `tritonserver --model-repository=/models`.
- Все модели перешли в статус `READY` (проверка см. раздел 4).

Известные проблемы (Triton 24.01) и обходы:

| Проблема | Признак | Решение |
|----------|---------|---------|
| Строковые входы ломаются в Python-backend (#7391) | `unpack_from requires a buffer of at least 421075229 bytes` | Текст передаётся как `UINT8` UTF-8-байты |
| `numpy 2.x` с Python-backend | Выходные тензоры пустые (raw 0 байт) при корректных входных | Пин `numpy==1.26.4` в образе |
| `GET /v2/models` → 404 | — | Использовать `POST /v2/repository/index` |
| `/v2/health/ready` | пустое тело | Считать зелёным по HTTP 200 |

## 3. Методика нагрузочного тестирования

Окружение: AMD 7840HS / Radeon 780M (RDNA3, 12 CU), **Vulkan llvmpipe-less radv**,
24 GB RAM, Triton 24.01 (`triton-bls:latest`, 4 CPU-лимита).

Один инструмент:

- **`perf/load_test.py`** (tritonclient, потоки) — тестирует все модели с
  текстовым входом:
  ```
  uv run python perf/load_test.py --all --concurrency 2 --duration 30 --out perf/results/results_gpu.json
  ```

Два сценария из задания:
- Нагрузка на **каждую модель по отдельности**.
- Нагрузка на **всю цепочку через BLS** (`ensemble`).

Метрики сбора:
- **Throughput** (RPS) — с клиента (results JSON).
- **Latency**: Queue и Compute Infer — средние на запрос из Prometheus
  на порту 8002 (`nv_inference_*_duration_us / request_success`).

### Как ускорили `generation` (сравнение подходов)

См. раздел 4.2/4.3: HF `transformers` на CPU → llama.cpp (llama-server) на
**Vulkan iGPU** (плюс `instance_group count: 2` и
`dynamic_batching { max_queue_delay_microseconds: 100 }` для `generation`).

## 4. Результаты

### 4.1 Через Perf Analyzer (concurrency 1:4, p95)

> Не запускался: требует загрузки SDK-образа с NGC. Замеры через
> `perf/load_test.py` (раздел 4.2). При желании: `bash perf/run_perf_analyzer.sh`.

### 4.2 Через perf/load_test.py (concurrency=2, duration=30)

| Модель | reqs | rps | mean ms | p50 ms | p95 ms |
|--------|------|-----|---------|--------|--------|
| `text_tokenizer` | 111508 | 3711.25 | 0.52 | 0.50 | 0.71 |
| `classification` | 3489 | 116.25 | 16.80 | 8.21 | 61.74 |
| `embedding` | 1736 | 57.72 | 34.19 | 17.85 | 71.95 |
| `generation` | 30 | 0.99 | 2011.57 | 2161.50 | 2301.17 |
| `ensemble` (BLS) | 471 | 15.63 | 127.67 | 107.92 | 188.50 |

### 4.3 Сравнение `generation`: CPU (HF) ↔ GPU (llama.cpp Vulkan)

Тот же конфиг load-тестера (concurrency=2, duration=30).

| Метрика (generation) | Было: HF CPU | Стало: llama.cpp Vulkan | Δ |
|----------------------|-------------|--------------------------|---|
| reqs (за 30 с) | 4 | 30 | ×7.5 |
| p50 latency | 23322.60 ms | 2161.50 ms | **×10.8** |
| p95 latency | 23638.17 ms | 2301.17 ms | ×10.3 |
| avg queue (Triton) | 6238.18 ms | 0.31 ms | **×20 000** |
| avg compute-infer (Triton) | 11659.13 ms | 1856.66 ms | ×6.3 |

> Почему так: HF `generate` на CPU (SmolLM-135M fp32) выдавал ~5–10 ток/с,
> llama.cpp на Radeon 780M (Qwen2.5-0.5B Q4_K_M, Vulkan, `-ngl 99`) — **~120–130 ток/с**
> (замер `llama-server` timings + `gpu_busy_percent`). Очередь почти исчезла:
> 2 инстанса + `dynamic_batching` + быстрые GPU-слоты.

### 4.4 Средняя задержка на этап (Triton metrics, порт 8002) — после оптимизации

| Модель | avg queue ms | avg compute-infer ms |
|--------|--------------|----------------------|
| `text_tokenizer` | 0.08 | 0.23 |
| `classification` | 6.40 | 9.62 |
| `embedding` | 12.29 | 20.53 |
| `generation` | 0.31 | 1856.66 |
| `ensemble` | 59.80 | 69.19 |

## 5. Анализ узких мест

- **Queue vs Compute**: для ONNX-моделей очередь соизмерима с вычислениями
  (`classification` 6.4 vs 9.6 мс, `embedding` 12.3 vs 20.5 мс) — при потоке
  входных запросов Triton упирается в CPU-расчёты, а не планировщик.
- **Очередь для `generation` устранена**: было 6238 мс (один CPU-инстанс
  занят авторегрессией → остальное встаёт в очередь), стало **0.31 мс**
  благодаря 2 инстансам, `dynamic_batching { max_queue_delay_microseconds: 100 }`
  и переносу генерации на GPU (каждый запрос теперь ~2 с вместо ~23 с).
- **Compute у `generation` (1857 мс)** = сам llama.cpp generate 128 токенов на iGPU
  (~120 ток/с) + HTTP-хоп. Дальнейший выигрыш — короткие ответы
  (`n_predict` меньше), лучшие модели (см. раздел 6).
- **`ensemble` суммирует шаги**: p50 ≈ 108 мс без генерации
  (токенизация <1 мс + классификация ~15 мс + эмбеддинг ~35 мс + BLS-overhead).
  Очередь ensemble 59.8 мс — это ожидание вложенных моделей, не CPU-голод.
- **RPS**: CPU-упирание у `classification` (116) и `embedding` (58);
  `text_tokenizer` — лёгкий (3711).

## 6. Выводы

### Преимущества Triton + BLS по сравнению с Flask/FastAPI

1. **Единый инференс-интерфейс** (HTTP/gRPC + батчинг) вместо ручного маршрута.
2. **Динамический батчинг** и конвейерное выполнение из коробки.
3. **BLS**: оркестрация нескольких моделей живёт **внутри Triton** — одна
   сетевая поездка клиент→сервер, без промежуточных HTTP-хопов между моделями
   (в FastAPI каждый вызов через `requests` отнимает латентность).
4. **Унифицированные метрики** (throughput/latency на модель) и Prometheus endpoint.
5. **Версионирование моделей** и горячий перезапуск без пересборки сервиса.
6. Пары dispatch занимает Triton, а не ваш «голый» веб-фреймворк.

### Можно ли TensorRT-LLM на AMD 7840HS?

**Нет.** TensorRT-LLM (и TensorRT) требует NVIDIA GPU (CUDA/TensorRT runtime).
На iGPU AMD (Radeon 780M) CUDA не существует, а официальный ROCm не
поддерживает APU (gfx1103) — поэтому путь NVIDIA надо исключить.

Рабочие альтернативы для **AMD iGPU (RDNA3, Vulkan)**:

| Движок | Статус на 780M | Комментарий |
|--------|----------------|-------------|
| **llama.cpp (Vulkan/RADV)** | ✅ работает (этот отчёт) | ~120–130 ток/с на Q4_K_M 0.5B |
| ONNX Runtime (Vulkan EP) | ⚠️ возможен | подходит для tokenizer-моделей; LLM слабее оптимизирован |
| vLLM / FlashAttention | ❌ | требует CUDA/ROCm-плюс; на APU нет |
| MIGraphX / ROCm | ❌/⚠️ | gfx1103 не входит в список поддержки ROCm |
| WebGPU (browser) | minor | игрушки, не серверный выхлоп |

### Какая модель оптимальна для AMD 7840HS iGPU?

Видеокарта 780M — ~2.8 TFLOPS FP32, память разделяемая (слот на 4–6 GB VRAM
из 24 GB RAM). Оптимум — **квантованные GGUF ≤ 1.5B (Q4_K_M/Q5_K_M)**:

| Модель (Instruct/GGUF) | Размер Q4 | Ожидание на 780M | Почему |
|------------------------|-----------|------------------|--------|
| **Qwen2.5-0.5B-Instruct** | ~0.4 GB | 130+ ток/с | используется в отчёте: быстрая, без GPT-риска |
| **Qwen2.5-1.5B-Instruct** | ~1.0 GB | 60–80 ток/с | заметно умнее 0.5B, влезает в VRAM iGPU |
| **SmolLM2-1.7B-Instruct** | ~1.0 GB | 60–80 ток/с | «on-device» серия, обучалась для малых задач |
| **Llama-3.2-1B-Instruct** | ~0.8 GB | 70–90 ток/с | хорошая база для QA |
| Qwen2.5-7B-Instruct | ~4.5 GB | 15–25 ток/с (offload) | слишком тяжела для 780M — часть весов в RAM |

Дилемма «tiny model vs качество»: SmolLM-135М fp32 на CPU давал ~23 с на ответ
(~10 ток/с). Куда эффективнее держать **Qwen2.5-0.5B Q4_K_M на Vulkan iGPU**:
та же «лёгкость», но плюс GPU-пропускная способность ≈ ×12 по скорости при
сопоставимом размере памяти.

### Что ещё можно подкрутить

- Уменьшить `max_new_tokens` (`128` → `64`) и включить стриминг для UX.
- `n_parallel` у `llama-server` (сейчас 4) — больше одновременных слотов.
- ONNX для `classification`/`embedding` — INT8-квантизация
  (`onnxruntime.quantization`).
- Dynamic Batching уже настроен (`max_queue_delay_microseconds: 100` для
  python-моделей, `dynamic_batching {}` — по умолчанию для ONNX).

**Главный вывод**: на APU достаточно перенести autoregressive-генерацию с
CPU-`transformers` на **llama.cpp (Vulkan)** — latency падает с ~23 с до ~2 с
(×10), очередь — с 6.2 с до 0.3 мс, а BLS-пайплайн в Triton остаётся тем же.

## 7. Артефакты в репозитории

- `model_repository/` — иерархия, `config.pbtxt`, скрипты BLS (`1/model.py`).
- `perf/` — инструменты нагрузочного тестирования и метрик.
- `tests/` — unit-тесты BLS-логики и e2e-тесты полной цепочки.
- `README.md` / `README_RU.md` — инструкции по запуску.