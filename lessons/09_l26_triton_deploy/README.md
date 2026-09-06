# Triton Inference Server: Model Orchestration (Homework)

Задание: [TASK.md](./TASK.md). Отчёт по результатам: [REPORT.md](./REPORT.md).
Русская версия: [README_RU.md](./README_RU.md).

## Context

Домашнее задание по оркестрации моделей в NVIDIA **Triton Inference Server**.
Нужно развернуть пайплайн из трёх разнородных лёгких моделей на разных бэкендах,
объединить их Business Logic Scripting (BLS) моделью, запустить в Docker,
провести нагрузочное тестирование и собрать отчёт.

## Architecture

| Model | Backend | Model | Params | Role |
|-------|---------|--------------------|--------|------|
| `text_tokenizer` | python | `huawei-noah/TinyBERT_General_4L_312D` tokenizer | — | Tokenization |
| `classification` | onnxruntime | TinyBERT 4L-312D (ONNX) | 14.5M | Question/statement classification |
| `embedding` | onnxruntime | `sentence-transformers/all-MiniLM-L6-v2` (ONNX) | 22.7M | 384-d embeddings |
| `generation` | python (HTTP) | llama.cpp (`llama-server`) + Qwen2.5-0.5B-Instruct-Q4_K_M.gguf | 0.5B | Answer generation on **Vulkan iGPU** |
| `ensemble` | python (BLS) | — | — | Orchestrates the chain |

```
Input Text
   │
   ▼
text_tokenizer ──► embedding (mean-pool + L2-norm)
   │
   └────────────► classification
                     │
               [statement] ─► return "[Classification] ..."
               [question]  ─► generation ─► return "[Generated] ..."
```

The BLS model (`ensemble`) calls the others internally via
`pb_utils.InferenceRequest` / `.exec()`, so the client makes **one** request
and receives the final pipeline result.

## Prerequisites

- Docker with Compose v2 (`docker compose version`)
- Access to `nvcr.io/nvidia/tritonserver` — **free NGC login required** for the pull (step 3 below)
- Python 3.10+ with [uv](https://docs.astral.sh/uv/)
- Roughly 8 GB RAM and 4+ CPU cores recommended
- Works **CPU-only except generation**: it accelerates on the AMD iGPU via
  **llama.cpp + Vulkan** (`/dev/dri`, Radeon 780M). No NVIDIA GPU required.
- For generation you need a running **llama.cpp `llama-server`** on the host
  (Vulkan build) — the Triton `generation` model calls it over HTTP
  (`LLAMA_SERVER_URL`, default `http://host.docker.internal:8080`).
  Without it, generation falls back to CPU `transformers`.

## Setup

### 1. Install Python dependencies (uv)

```bash
uv sync
```

Installs the lockfile with CPU-only PyTorch (`torch` pulled from the
`pytorch-cpu` index — no CUDA wheels).

### 2. Export the ONNX models

```bash
uv run python export_models.py
```

Downloads TinyBERT + MiniLM, exports both to ONNX and writes them into
`model_repository/{classification,embedding}/1/`. The `text_tokenizer` and
`ensemble` models are Python-backend and need no export. The `generation`
model serves a **GGUF** via llama.cpp: put `model.gguf`
(e.g. `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, ~0.4 GB) into
`model_repository/generation/1/model.gguf`.

> **Note on the classification model.** TinyBERT is a *generic pretrained*
> checkpoint — its classification head is randomly initialized and, without
> training, returns near-zero logits (confidence ~0.00x, i.e. noise). The
> classifier used in this homework was therefore **fine-tuned on a small
> ~290-sample statement/question dataset** (3 epochs, CPU). Re-run it anytime:
>
> ```bash
> uv run python train_classifier.py     # fine-tunes + exports ONNX to model_repository/classification/1/
> docker compose restart                # reload the model in Triton
> ```
>
> After training you get meaningful probabilities (e.g. `confidence=0.707`)
> instead of coin-flip logits. The raw `export_models.py` output is **not**
> meant to be used as-is for classification.

### 3. Pull the Triton image (NGC)

The image `nvcr.io/nvidia/tritonserver:24.01-py3` lives on NVIDIA's NGC
registry, which blocks anonymous pulls (**403 Forbidden**). A free account is
enough:

1. Register at <https://ngc.nvidia.com/signup> (free).
2. Generate an API key: <https://ngc.nvidia.com/setup/api-key>.
3. Log in — username is always `$oauthtoken`, password is the API key:

   ```bash
   docker login nvcr.io
   # Username: $oauthtoken
   # Password: <NGC API key>
   ```

4. Verify the pull works before starting compose:

   ```bash
   docker pull nvcr.io/nvidia/tritonserver:24.01-py3
   ```

   If you only have access to a *different* Triton image, override the tag
   without touching the repo:
   `TRITON_IMAGE=<your-image> docker compose up -d`. The same applies to the
   SDK image used by Perf Analyzer via `SDK_IMAGE=<...> ./perf/run_perf_analyzer.sh`.

## Run

### Start llama.cpp `llama-server` (Vulkan on the AMD iGPU) — one-time

The `generation` model answers over HTTP from a local llama.cpp server. Start
it on the host (binary must be a **Vulkan** build):

```bash
bash perf/start_llama_server.sh
# curl http://127.0.0.1:8080/health   → {"status":"ok"}
```

Defaults: model `model_repository/generation/1/model.gguf`, `-ngl 99`
(offload all layers to the iGPU), port `8080`, `-np 4` parallel slots.
Override via env, e.g. `PORT=8090 MODEL=/path/to/model.gguf`.

### Start Triton (Docker)

```bash
docker compose build      # installs CPU torch, transformers, numpy==1.26.4
docker compose up -d
```

- mounts `./model_repository` to `/models`
- maps ports **8000** (HTTP), **8001** (gRPC), **8002** (metrics)
- runs `tritonserver --model-repository=/models`
- `extra_hosts` maps `host.docker.internal` → host so the `generation` model
  can reach `llama-server` (`LLAMA_SERVER_URL` env override supported)

### Stop

```bash
docker compose down
```

## Manual verification checklist (do in this order)

1. **Health check** — server is up:
   ```bash
   curl -s http://localhost:8000/v2/health/live && echo
   curl -s -w "%{http_code}\n" -o /dev/null http://localhost:8000/v2/health/ready
   ```
   Both are HTTP 200 when ready (Triton 24.01 returns an empty body, not `1`).

2. **All 5 models READY** — wait for `"READY"` state (first prompt evaluation in
   llama.cpp warms up shader/GPU state; can take a few seconds):
   ```bash
   curl -s -X POST http://localhost:8000/v2/repository/index
   ```
   Expected: `text_tokenizer`, `classification`, `embedding`, `generation`,
   `ensemble` — all `state: "READY"`. (Note: `GET /v2/models` returns 404 in
   Triton 24.01.)

3. **Smoke-test the whole chain in one request**:
   ```bash
   uv run python client.py "What is machine learning?"
   # → "[Generated] ..."   (question routes to generation)
   uv run python client.py "I love this product."
   # → "[Classification] label=..., confidence=..."
   ```

4. **Run the test suites** (see below).

5. **Load testing** (see below) — Perf Analyzer and/or `perf/load_test.py`.

6. **Check Triton metrics on port 8002**:
   ```bash
   curl http://localhost:8002/metrics | head
   uv run python perf/metrics.py
   ```

## Tests

### 1. Unit tests (offline — no Triton server needed)

Exercises the BLS routing logic against a mock `pb_utils`:

```bash
uv run pytest tests/ -v
```

Covers: statement→classification routing, question→generation routing,
embedding/tokenizer are invoked, generation receives the prompt, batch
handling, missing-model error.

### 2. E2E tests (require a live Triton server)

```bash
# against an already-running server
uv run pytest tests/test_e2e.py -v

# or let pytest start the docker stack and tear it down afterwards
TRITON_START_DOCKER=1 uv run pytest tests/test_e2e.py -v
```

Covers: server live, all 5 models reach READY, a single `ensemble` request
returns a valid response, direct inference on `generation` and the
tokenizer→classifier / tokenizer→embedding chains. Tests **skip cleanly**
when no server is reachable.

### 3. Integration script (tritonclient, against a live server)

```bash
uv run python test_ensemble.py
```

Per-model checks plus the full BLS pipeline over gRPC.

## Load testing (Part 4)

### NVIDIA Perf Analyzer (SDK image)

```bash
./perf/run_perf_analyzer.sh
```

Runs `perf_analyzer` from the Triton SDK container against
`text_tokenizer`, `embedding`, `generation`, `ensemble`
(concurrency 1:4, p95, 5 s windows). `classification` needs pre-tokenized
input — use the Python load tester below instead.

### Python load tester (no SDK needed)

```bash
uv run python perf/load_test.py --model ensemble --concurrency 4 --duration 20
uv run python perf/load_test.py --all --concurrency 2 --duration 10 --out perf/results/results.json
```

Reports RPS and mean/p50/p95/p99 latencies per model. Tokenized models
(`classification`, `embedding`) are fed via the local tokenizer.

### Latency breakdown (Triton metrics)

```bash
uv run python perf/metrics.py
```

Shows per-model inference counts, Queue and Compute Infer latency sums from
the Prometheus endpoint on port 8002 — use it to find bottlenecks.

## Report (Part 5)

See `REPORT.md` — includes repo tree, configs summary, methodology, result
tables (fill with your measurements), bottleneck analysis, and conclusions
(Triton/BLS vs Flask/FastAPI, optimization options).

## Repository layout

```
├── client.py                  # single-request BLS client
├── docker-compose.yaml        # Triton server (ports 8000/8001/8002)
├── Dockerfile                 # repo image with transformers/optimum
├── export_models.py           # ONNX export for classification/embedding
├── train_classifier.py        # fine-tune TinyBERT head + export ONNX
├── model_repository/          # all models with config.pbtxt + version dirs
├── perf/                      # load testing tools
├── perf/start_llama_server.sh # llama.cpp llama-server (Vulkan iGPU) launcher
├── perf/results/              # (generated load-test results, gitignored)
├── pyproject.toml             # uv project / deps
├── reference/                 # reference solution from the task
├── REPORT.md                  # homework report template
├── test_ensemble.py           # gRPC integration checks
├── tests/                     # pytest: unit (offline) + e2e (live server)
└── README_RU.md / README.md
```

## Notes / troubleshooting

- **Image pull denied (403)** for `nvcr.io` — complete the NGC login from
  "Pull the Triton image" step 3, or reuse a Triton image you can access:
  `TRITON_IMAGE=<your-image> docker compose up -d` (same for the Perf Analyzer
  SDK image via `SDK_IMAGE`).
- **Model stuck NOT READY** — watch `docker compose logs -f triton`; check that
  `bash perf/start_llama_server.sh` is running (else generation falls back to
  slow CPU `transformers`, ~23 s per answer).
- **Triton 24.01 + `numpy 2.x`** — the Python backend returns empty output
  tensors (0 raw bytes) unless `numpy==1.26.4` is pinned in the image;
  the Dockerfile already does this.
- **String (`TYPE_STRING`) inputs fail** on Triton 24.01 Python backend
  (issue #7391). Textual `TEXT` is therefore exchanged as UTF-8 bytes in a
  `UINT8` tensor (see `client.py`/`tests`); don't switch back to `BYTES`
  without upgrading Triton.
- **CPU-only** is the default; to use an NVIDIA GPU set `kind: KIND_GPU` in each
  `config.pbtxt` and remove CPU/memory limits from compose.
- Output tensor shapes are dynamic (`dims: [-1]` / `[-1, 128]`), padded to
  `max_length=128`; no fixed-size assumptions in configs or clients.

## License / scope

Homework project for Triton Inference Server practice. NVIDIA GPU is *not*
required — the entire stack runs on CPU (also compatible with AMD iGPU).