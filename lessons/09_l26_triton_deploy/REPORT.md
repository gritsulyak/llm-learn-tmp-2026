# Отчёт: оркестрация моделей в NVIDIA Triton Inference Server

## 1. Что настроено

Развёрнут пайплайн из трёх разнородных лёгких моделей, объединённых
Business Logic Scripting (BLS) моделью на Python backend:

| Модель | Бэкенд | Модель | Параметры | Роль |
|--------|--------|--------|-----------|-----|
| `text_tokenizer` | python | TinyBERT tokenizer | — | Токенизация текста |
| `classification` | onnxruntime | TinyBERT 4L-312D | 14.5M | Двоичная классификация (вопрос/утверждение) |
| `embedding` | onnxruntime | all-MiniLM-L6-v2 | 22.7M | Эмбеддинги 384-d (mean-pooling + L2-norm) |
| `generation` | python (HTTP) | **llama.cpp (llama-server)**, Qwen2.5-0.5B-Instruct Q4_K_M | 0.5B | Генерация ответа на вопрос (llama.cpp: **CPU** \| **Vulkan iGPU**) |
| `ensemble` | python (BLS) | — | — | Дирижёр: токенизация → эмбеддинг + классификация → генерация |

`generation` уже **не крутит HF-трансформер на CPU**: через Python-backend он
делает HTTP-вызов к локальному `llama-server` (llama.cpp), который грузит
GGUF-модель. Бэкенд llama.cpp переключаемый: **Vulkan iGPU** (Radeon 780M,
`-ngl 99`) или чисто **CPU**-сборка (ggml-cpu, без Vulkan) — прогоны обоих
вариантов в разделе 4. Если llama-server недоступен — фолбэк на
`transformers` (CPU).

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
  текстовым входом (для CPU- и Vulkan-прогонов llama.cpp соответственно):
  ```
  uv run python perf/load_test.py --all --concurrency 2 --duration 30 --out perf/results/results_cpu.json
  uv run python perf/load_test.py --all --concurrency 2 --duration 30 --out perf/results/results_vulkan.json
  ```

Два сценария из задания:
- Нагрузка на **каждую модель по отдельности**.
- Нагрузка на **всю цепочку через BLS** (`ensemble`).

Метрики сбора:
- **Throughput** (RPS) — с клиента (results JSON).
- **Latency**: Queue и Compute Infer — средние на запрос из Prometheus
  на порту 8002 (`nv_inference_*_duration_us / request_success`).

### Как настраивали `generation` (сравнение бэкендов)

См. раздел 4.2/4.3: llama.cpp (llama-server) — переключаемый бэкенд
**CPU** (ggml-cpu, без Vulkan) ↔ **Vulkan iGPU** (плюс
`instance_group count: 2` и `dynamic_batching { max_queue_delay_microseconds: 100 }`
для `generation`).

## 4. Результаты

### 4.1 Через Perf Analyzer (concurrency 1:4, p95)

> Не запускался: требует загрузки SDK-образа с NGC. Замеры через
> `perf/load_test.py` (раздел 4.2). При желании: `bash perf/run_perf_analyzer.sh`.

### 4.2 Через perf/load_test.py (concurrency=2, duration=30)

Оба прогона сделаны на **одном и том же состоянии** модели (GGUF
Qwen2.5-0.5B Q4_K_M) и кода BLS, с интервалом в один запуск; различается
только бэкенд `llama-server` — CPU-сборка (ggml-cpu, без Vulkan) или
Vulkan iGPU (`-ngl 99`).

Прогон 1 — llama.cpp **CPU** (без Vulkan) — `perf/results/results_cpu.json`:

| Модель | reqs | rps | mean ms | p50 ms | p95 ms |
|--------|------|-----|---------|--------|--------|
| `text_tokenizer` | 124662 | 4149.01 | 0.47 | 0.45 | 0.59 |
| `classification` | 3341 | 111.15 | 17.58 | 8.77 | 62.67 |
| `embedding` | 1343 | 44.61 | 44.19 | 24.82 | 77.54 |
| `generation` | 22 | 0.72 | 2765.56 | 2703.09 | 3685.14 |
| `ensemble` (BLS) | 23 | 0.72 | 2706.10 | 2761.87 | 2814.34 |

Прогон 2 — llama.cpp **Vulkan iGPU** (Radeon 780M, `-ngl 99`) —
`perf/results/results_vulkan.json`:

| Модель | reqs | rps | mean ms | p50 ms | p95 ms |
|--------|------|-----|---------|--------|--------|
| `text_tokenizer` | 125692 | 4183.51 | 0.46 | 0.45 | 0.58 |
| `classification` | 3362 | 111.92 | 17.45 | 8.47 | 62.55 |
| `embedding` | 1599 | 53.17 | 37.00 | 20.02 | 76.03 |
| `generation` | 36 | 1.13 | 1776.28 | 1948.93 | 2789.89 |
| `ensemble` (BLS) | 31 | 0.98 | 2013.81 | 2014.47 | 2060.69 |

> В обоих прогонах классификатор относит тестовый текст к классу «вопрос»,
> поэтому `ensemble` идёт через шаг `generation` (счётчик Triton
> `generation` = прямой вызов + вызовы из BLS: 22+23 CPU / 36+31 Vulkan).
> Условия прогонов идентичны — сравнение корректно.

### 4.3 Сравнение `generation`: llama.cpp CPU ↔ llama.cpp Vulkan

Тот же конфиг load-тестера (concurrency=2, duration=30), то же состояние
модели и кода; различается только бэкенд llama.cpp: **ggml-cpu без Vulkan**
(`-t 4`, 4 потока) против **Vulkan iGPU Radeon 780M** (`-ngl 99`).

| Метрика (generation) | llama.cpp CPU | llama.cpp Vulkan | Δ CPU→Vulkan |
|----------------------|---------------|------------------|--------------|
| reqs (за 30 с) | 22 | 36 | ×1.64 |
| rps | 0.72 | 1.13 | ×1.57 |
| p50 latency | 2703.09 ms | 1948.93 ms | **×1.39** |
| p95 latency | 3685.14 ms | 2789.89 ms | ×1.32 |
| avg queue (Triton) | 0.26 ms | 0.26 ms | — |
| avg compute-infer (Triton) | 2047.88 ms | 1416.14 ms | ×1.45 |

Прямой замер токенов/с тем же GGUF (llama-bench, build `0b1bad14f`/10380,
Qwen2.5-0.5B Q4_K_M, 494M) на той же машине:

| Backend | pp512, ток/с | tg128, ток/с |
|---------|-------------|--------------|
| CPU (`-t 4`, как в start_llama_server.sh) | 543.96 ± 4.30 | 99.13 ± 0.23 |
| CPU (`-t 8`) | 886.71 ± 39.74 | 105.52 ± 0.22 |
| Vulkan iGPU Radeon 780M (`-ngl 99`) | 4604.77 ± 76.01 | 123.28 ± 5.26 |

> Vulkan-бэкенд даёт **~25% к генерации** (123 vs 99 ток/с на 4 потоках —
> авторегрессия упирается в память/мелкий decode), зато **prefill
> ускоряется в ~8.5 раз** (4605 vs 544 ток/с), что важно при длинных
> промптах. Поэтому p50 `generation` через Triton падает с ≈2.70 с (CPU) до
> ≈1.95 с (Vulkan), а rps растёт 0.72→1.13. Очередь минимальна в обоих
> вариантах: 2 инстанса + `dynamic_batching` + быстрые слоты.

### 4.4 Средняя задержка на этап (Triton metrics, порт 8002)

Прогон 1 — llama.cpp **CPU** (без Vulkan):

| Модель | avg queue ms | avg compute-infer ms |
|--------|--------------|----------------------|
| `text_tokenizer` | 0.08 | 0.20 |
| `classification` | 7.81 | 8.76 |
| `embedding` | 20.07 | 21.67 |
| `generation` | 0.26 | 2047.88 |
| `ensemble` | 1322.41 | 1382.84 |

Прогон 2 — llama.cpp **Vulkan iGPU**:

| Модель | avg queue ms | avg compute-infer ms |
|--------|--------------|----------------------|
| `text_tokenizer` | 0.08 | 0.20 |
| `classification` | 7.61 | 8.71 |
| `embedding` | 16.42 | 18.32 |
| `generation` | 0.26 | 1416.14 |
| `ensemble` | 989.09 | 1023.36 |

> В обоих прогонах `ensemble` ждал вложенный `generation` (класс «вопрос»):
> очередь ~990–1320 мс — это ожидание 2 инстансов generation, а не CPU-голод
> BLS. Compute-время `generation` = generate 128 токенов + HTTP-хоп.

## 5. Анализ узких мест

- **Queue vs Compute**: для ONNX-моделей очередь соизмерима с вычислениями
  (`classification` ~7.6–7.8 vs ~8.7–8.8 мс, `embedding` 16.4–20.1 vs 18.3–21.7 мс)
  — при потоке входных запросов Triton упирается в CPU-расчёты, а не планировщик.
- **Очередь для `generation` минимальна в обоих бэкендах (~0.26 мс)**:
  2 инстанса + `dynamic_batching { max_queue_delay_microseconds: 100 }` и быстрые
  слоты llama.cpp — авторегрессия одного запроса не блокирует остальные.
- **Compute у `generation` (1416 мс на Vulkan / 2048 мс на CPU)** = генерация
  128 токенов (123 ток/с на iGPU vs ~99 ток/с на CPU) + HTTP-хоп. Дальнейший
  выигрыш — короткие ответы (`n_predict` меньше), лучшие модели (см. раздел 6).
- **`ensemble` упирается в `generation`**: в обоих прогонах тестовый текст —
  класс «вопрос», поэтому очередь `ensemble` (989–1322 мс) — это ожидание
  вложенного `generation` (2 инстанса), а не узкое место BLS/CPU-голод.
- **RPS**: `generation` — 0.72 (CPU) → 1.13 (Vulkan), `classification` — ~112,
  `embedding` — 45–53; `text_tokenizer` — лёгкий (~4.2k).

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
| **llama.cpp (CPU / Vulkan)** | ✅ работает (этот отчёт) | ~99–105 ток/с (CPU) / ~120–130 ток/с (Vulkan) на Q4_K_M 0.5B |
| ONNX Runtime (Vulkan EP) | ⚠️ возможен | подходит для tokenizer-моделей; LLM слабее оптимизирован |
| vLLM / FlashAttention | ❌ | требует CUDA/ROCm-плюс; на APU нет |
| MIGraphX / ROCm | ❌/⚠️ | gfx1103 не входит в список поддержки ROCm |
| WebGPU (browser) | minor | игрушки, не серверный выхлоп |

### Какая модель оптимальна для AMD 7840HS iGPU?

Видеокарта 780M — ~2.8 TFLOPS FP32, память разделяемая (слот на 4–6 GB VRAM
из 24 GB RAM). Оптимум — **квантованные GGUF ≤ 1.5B (Q4_K_M/Q5_K_M)**:

| Модель (Instruct/GGUF) | Размер Q4 | Ожидание на 780M | Почему |
|------------------------|-----------|------------------|--------|
| **Qwen2.5-0.5B-Instruct** | ~0.4 GB | ~100 (CPU) / 130+ (iGPU) ток/с | используется в отчёте: быстрая, без GPT-риска |
| **Qwen2.5-1.5B-Instruct** | ~1.0 GB | 60–80 ток/с | заметно умнее 0.5B, влезает в VRAM iGPU |
| **SmolLM2-1.7B-Instruct** | ~1.0 GB | 60–80 ток/с | «on-device» серия, обучалась для малых задач |
| **Llama-3.2-1B-Instruct** | ~0.8 GB | 70–90 ток/с | хорошая база для QA |
| Qwen2.5-7B-Instruct | ~4.5 GB | 15–25 ток/с (offload) | слишком тяжела для 780M — часть весов в RAM |

«Tiny model vs качество»: для 7840HS достаточно **llama.cpp на любой CPU-сборке** —
Qwen2.5-0.5B Q4_K_M выдаёт ~99–105 ток/с уже на CPU. Подключение **Vulkan iGPU**
ускоряет генерацию ещё на ~25% (до ~123 ток/с) и даёт **×8.5 к prefill**, что
критично при длинных промптах.

### Что ещё можно подкрутить

- Уменьшить `max_new_tokens` (`128` → `64`) и включить стриминг для UX.
- `n_parallel` у `llama-server` (сейчас 4) — больше одновременных слотов.
- ONNX для `classification`/`embedding` — INT8-квантизация
  (`onnxruntime.quantization`).
- Dynamic Batching уже настроен (`max_queue_delay_microseconds: 100` для
  python-моделей, `dynamic_batching {}` — по умолчанию для ONNX).

**Главный вывод**: переход генерации на **llama.cpp** снимает узкое место
авторегрессии. На чистой (без-Vulkan) CPU-сборке p50 `generation` ≈ 2.7 с;
подключение **Vulkan iGPU** дожимает до ≈1.95 с (×1.39). Очередь — <0.3 мс в
обоих вариантах, а BLS-пайплайн в Triton остаётся тем же.

## 7. Артефакты в репозитории

- `model_repository/` — иерархия, `config.pbtxt`, скрипты BLS (`1/model.py`).
- `perf/` — инструменты нагрузочного тестирования и метрик.
- `tests/` — unit-тесты BLS-логики и e2e-тесты полной цепочки.
- `README.md` / `README_RU.md` — инструкции по запуску.