# AGENTS.md

## Project Overview

**DSL Query Generator** — домашнее задание курса «LLM-инженер» (занятие 21, LLM-judge).
Дообучаем малую открытую LLM переводить русские запросы менеджеров в корпоративный
язык запросов **ECQL** компании E-Corp (вымышленный DSL). Датасет генерирует
модель-учитель **YandexGPT Pro** (`yandexgpt`, Yandex Cloud). Обучение — LoRA (r=4)
через `trl.SFTTrainer` на **CPU** (Qwen2.5-1.5B-Instruct). Оценка — метрики
синтаксиса/логики + **LLM-as-a-judge** (YandexGPT Pro). Отдельный ноутбук для
Google Colab GPU (Qwen3-4B-Instruct, QLoRA 4-bit).

## Stack

- Python 3.12+, managed with `uv` (CPU-сборка torch из `download.pytorch.org/whl/cpu`)
- `transformers==4.49.0`, `trl==0.16.0`, `peft==0.14.0`, `accelerate==1.3.0`,
  `datasets==3.3.0` — зафиксированы (совместимы друг с другом и с Colab-ноутбуком)
- Yandex Cloud Foundation Models API (`requests`), ключи в `.env` (копия из `../06_*`)
- `jupytext`/`nbformat`/`nbclient` — прогон процент-ноутбуков в `.ipynb`

## Key Commands

```bash
uv sync --group dev

# 1) Генерация датасета моделью-учителем (YandexGPT Pro) -> data/*.jsonl
uv run python scripts/generate_dataset.py --force --target 200

# 2) Валидация датасета (парсер ECQL, дубликаты, покрытие)
uv run python scripts/validate_dataset.py

# 3) Обучение LoRA на CPU (Qwen2.5-1.5B, r=4, alpha=8)
uv run python scripts/train_local.py --epochs 5

# 4) Оценка на тестовой выборке (метрики + LLM-as-a-judge)
uv run python scripts/evaluate.py            # с судьёй YandexGPT Pro
uv run python scripts/evaluate.py --skip-judge

# 5) Отчёт-ноутбук (исполненный .ipynb)
uv run python scripts/run_notebook.py --report notebooks/report_local.py \
    --output notebooks/report_local_run.ipynb

# 6) Тесты и линтер
uv run pytest -q
uv run ruff check src/ scripts/ tests/ notebooks/
```

## Project Structure

```
config/config.yaml        Общий конфиг (dataset, teacher, local_train, gpu_train, judge, eval)
src/ecql/
  config.py               Пути, загрузка .env и config.yaml
  ecql_spec.py            Грамматика ECQL: парсер + валидатор (FETCH/[]/@/&&/||/AS)
  yandex_client.py        YandexGPT Pro: учитель (генерация датасета) + судья
  dataset.py              Генерация/валидация/split датасета, stats
  train.py                LoRA-обучение (SFTTrainer), маскирование промпта, лог loss в CSV
  eval.py                 Метрики на тесте + LLM-as-a-judge + генерация ответов
scripts/                  CLI: generate/validate/train/evaluate/run_notebook
notebooks/
  report_local.py         Отчётный ноутбук (CPU): данные -> loss-кривая -> метрики -> judge
  train_colab.py          Ноутбук для Google Colab GPU (Qwen3-4B-Instruct, QLoRA)
  report_local_run.ipynb  Исполненный отчёт (результаты прогона)
data/                     ecql_dataset.jsonl / ecql_train.jsonl / ecql_test.jsonl
results/                  loss_log.csv, loss_curve.png, eval_local.json,
                          train_local.json, dataset_stats.json, lora_adapter_local/
tests/test_ecql.py        Pytest: грамматика, сплит, учитель/судья
```

## Code Conventions

- Пакет в `src/ecql/` (hatch layout, editable install через `uv sync`)
- Весь код и комментарии — на русском
- Обучение/инференс по умолчанию на CPU (fp32); GPU — только в `notebooks/train_colab.py`
- LoRA-параметры вынесены в `config/config.yaml`, не дублируются в коде
- Не добавлять комментарии, если они не требуются по заданию

## Before Committing

1. `uv run pytest -q` — все тесты зелёные
2. `uv run ruff check src/ scripts/ tests/ notebooks/` — без ошибок
3. `.env` не коммитить (содержит ключи Yandex Cloud)
4. `results/lora_adapter_local/` и `notebooks/*_run.ipynb` — в `.gitignore`