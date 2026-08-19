# %% [markdown]
# # Отчёт: DSL Query Generator (ECQL) — дообучение LLM на CPU
#
# ДЗ курса «LLM-инженер»: разработка генератора запросов на внутреннем языке ECQL.
# Дообучение малой модели `Qwen2.5-1.5B-Instruct` (LoRA r=4, alpha=8) на CPU,
# генерация датасета моделью-учителем YandexGPT Pro, оценка LLM-as-a-judge.
#
# Каждый блок `# %%` — отдельная ячейка. Запуск: `uv run python notebooks/report_local.py`
# или через `scripts/run_notebook.py`.

# %%
# 0. Конфигурация и пути

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ecql.config import (  # noqa: E402
    ADAPTER_DIR,
    DATA_DIR,
    EVAL_RESULTS,
    LOSS_LOG,
    RESULTS_DIR,
    load_yaml,
)
from ecql.dataset import dataset_stats, load_jsonl  # noqa: E402

torch.set_num_threads(16)

print(f"Проект: {ROOT}")
print(f"Время начала отчёта: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ## 1. Датасет
#
# 201 пара «русский запрос -> ECQL», сгенерирована моделью-учителем YandexGPT Pro
# (`yandexgpt` через Yandex Cloud Foundation Models). Разбиение 80/20 (160 train / 41 test).

# %%
train_rows = load_jsonl(DATA_DIR / "ecql_train.jsonl")
test_rows = load_jsonl(DATA_DIR / "ecql_test.jsonl")
full_rows = load_jsonl(DATA_DIR / "ecql_dataset.jsonl")
print(f"Всего пар: {len(full_rows)} | Train: {len(train_rows)} | Test: {len(test_rows)}")

for k, v in dataset_stats(full_rows).items():
    print(f"  {k}: {v}")

# %%
print("Примеры из датасета:")
for r in train_rows[:5]:
    print(f"  RU : {r['input']}")
    print(f"  ECQ: {r['output']}\n")

# %% [markdown]
# ## 2. Кривая обучения (Loss Curve)
#
# Обучение: CPU, fp32, LoRA r=4 alpha=8 на q/k/v/o проекциях, 160 примеров,
# batch=1, gradient accumulation=4 (эффективный batch=4), lr=2e-4 (cosine),
# 5 эпох, AdamW. Время на CPU: ~7 мин/эпоху на 16 потоках Ryzen 7 7840HS.

# %%
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

steps, losses, secs = [], [], []
with LOSS_LOG.open("r", encoding="utf-8") as fh:
    next(fh)  # header
    for line in fh:
        s, loss, sec, _ = line.strip().split(",")
        steps.append(int(s))
        losses.append(float(loss))
        secs.append(float(sec))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(steps, losses, marker="o", ms=3)
ax1.set_xlabel("шаг (оптимизатора)")
ax1.set_ylabel("Loss")
ax1.set_title("Loss по шагам (CPU)")
ax1.grid(alpha=0.3)
ax2.plot(secs, losses, marker="o", ms=3)
ax2.set_xlabel("время, сек")
ax2.set_ylabel("Loss")
ax2.set_title("Loss по времени")
ax2.grid(alpha=0.3)
fig.tight_layout()
loss_png = RESULTS_DIR / "loss_curve.png"
fig.savefig(loss_png, dpi=130)
print(f"График сохранён: {loss_png}")
print(f"Шагов: {len(steps)} | старт loss={losses[0]:.3f} | финиш loss={losses[-1]:.3f}")

# %% [markdown]
# ## 3. Сводка обучения
#
# Финальный LoRA-адаптер сохранён в `results/lora_adapter_local/` (4.4 МБ при r=4).

# %%
train_json = RESULTS_DIR / "train_local.json"
if train_json.exists():
    summary = json.loads(train_json.read_text(encoding="utf-8"))
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")

# %% [markdown]
# ## 4. Оценка на тестовой выборке (41 пример)
#
# Метрики: синтаксическая корректность (парсер ECQL), точное совпадение,
# совпадение сущности/полей/операторов, доля SQL-галлюцинаций, LLM-as-a-judge.

# %%
from ecql.eval import evaluate_on_testset, print_eval_summary  # noqa: E402
from ecql.train import load_adapter_and_tokenizer  # noqa: E402
from ecql.yandex_client import YandexGPTClient  # noqa: E402

cfg = load_yaml()
model, tokenizer = load_adapter_and_tokenizer(cfg["local_train"]["base_model"], ADAPTER_DIR, device="cpu")
model.eval()
print("Модель + адаптер загружены")

judge = YandexGPTClient(
    model=cfg["judge"]["model"],
    temperature=cfg["judge"]["temperature"],
    max_tokens=cfg["judge"]["max_tokens"],
)

t0 = time.time()
result = evaluate_on_testset(
    model,
    tokenizer,
    test_rows,
    max_new_tokens=cfg["eval"]["max_new_tokens"],
    device="cpu",
    judge=judge,
    use_judge=True,
)
print(f"Оценка заняла: {round(time.time() - t0, 1)} c")
print_eval_summary(result)

# %% [markdown]
# ## 5. Сравнительная таблица: вход — эталон — ответ модели
#
# 5 примеров из тестовой выборки (по заданию).

# %%
def print_comparison(result, n=5):
    preds = result["predictions"]
    header = f"{'#':<3}{'Вход (рус.)':<45}{'Эталон ECQL':<55}{'Ответ модели':<55}{'Синт.'}"
    print(header)
    print("-" * 210)
    for i, p in enumerate(preds[:n], 1):
        print(
            f"{i:<3}{p['input'][:44]:<45}{p['reference'][:54]:<55}"
            f"{p['prediction'][:54]:<55}{'✓' if p['syntax_valid'] else '✗'}"
        )

print_comparison(result)

# %% [markdown]
# ## 6. LLM-as-a-judge (YandexGPT Pro)
#
# Средний балл судьи и примеры вердиктов.

# %%
from collections import Counter  # noqa: E402

judge_scores = result["judge_scores"]

verdicts = Counter(s["verdict"] for s in judge_scores)
print("Вердикты судьи:", dict(verdicts))
print(f"Средний балл: {result['metrics']['judge_avg_score']} / 5")
print(f"Доля score>=4: {result['metrics']['judge_pass_rate']}")
print()
print("Примеры вердиктов судьи:")
for s in judge_scores[:5]:
    print(f"  score={s['score']} [{s['verdict']}] {s['comment'][:70]}")

# %% [markdown]
# ## 7. Анализ ошибок
#
# 1-2 примера, где модель ошиблась, с объяснением причины.

# %%
preds = result["predictions"]
bad = [p for p in preds if not p["syntax_valid"]]
for p in bad[:2]:
    print(f"  Вход : {p['input']}")
    print(f"  Эталон: {p['reference']}")
    print(f"  Ответ : {p['prediction']}")
    print()

# %% [markdown]
# ## 8. Выводы
#
# - Модель выучила каркас языка: `FETCH [ENTITY] WHERE ... && ... AS ...`
# - Сложнее всего даётся замена SQL-привычек: `>`, `<`, `AND` вместо `ABOVE/BELOW/&&`
#   (галлюцинации «в сторону SQL»), также встречаются выдуманные операторы и поля.
# - LLM-as-judge (YandexGPT Pro) подтверждает: большинство ответов семантически близки
#   эталону (средний балл 3.7-4.0 / 5), но синтаксическая точность требует
#   больше данных и/или более сильной модели (см. `notebooks/train_colab.py`).

# %%
print("Отчёт завершён.")
print(f"Результаты сохранены: {EVAL_RESULTS}")
