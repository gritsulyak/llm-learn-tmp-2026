# %% [markdown]
# # Google Colab (GPU): дообучение Qwen3-4B-Instruct на ECQL — QLoRA
#
# Отдельный ноутбук для запуска на облачном GPU (Google Colab, T4 16 ГБ).
# Локальная версия — `notebooks/report_local.py` (CPU, Qwen2.5-1.5B).
#
# **Выбор модели.** Из трёх рекомендованных кандидатов — **Phi-4 (14B)**,
# **Mistral 3 / Small 3 (24B)** и **Qwen3-4B-Instruct (4B)** — взят
# `Qwen/Qwen3-4B-Instruct`:
# 1. **Влезает в T4 16 ГБ** даже с 4-битным QLoRA (~3 ГБ весов). Phi-4 (14B) и
#    Mistral 3 (24B) в 4-битном виде занимают ~8 и ~13 ГБ и не помещаются вместе
#    с оптимизатором/градиентами;
# 2. **Сильный мультиязычный (включая русский) трейнинг** — вход у нас русский язык;
#    Phi-4 ориентирована на английский, у Mistral 3 русский заметно слабее;
# 3. **Преемственность с локальным экспериментом** — семейство Qwen (Qwen2.5-1.5B
#    локально и Qwen3-4B здесь) даёт одинаковые chat-шаблоны и LoRA-таргеты.
#
# Везде замеряется время (обучение, инференс) — сравнивайте с CPU-версией.

# %% [markdown]
# ## 0. Установка зависимостей
#
# Версии зафиксированы, чтобы совпадать с локальным CPU-конфигом (`pyproject.toml`).

# %%
# %%capture
!pip install -q transformers==4.49.0 trl==0.16.0 peft==0.14.0 accelerate==1.3.0 \
    datasets==3.3.0 bitsandbytes

import torch, time, json, sys
from pathlib import Path
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Конфигурация
#
# Параметры совпадают с секцией `gpu_train` в `config/config.yaml` проекта.

# %%
TCFG = {
    "base_model": "Qwen/Qwen3-4B-Instruct",
    "max_seq_length": 512,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2.0e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "lora_r": 4,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "optim": "paged_adamw_8bit",
}
MAX_NEW_TOKENS = 128

# %% [markdown]
# ## 2. Данные
#
# Загрузите в `/content/data/` файлы `ecql_train.jsonl` и `ecql_test.jsonl`
# (они лежат в `data/` этого репозитория — результат ДЗ, см. локальный отчёт).
# Вспомогательные функции определим прямо здесь, чтобы ноутбук был самодостаточным.

# %%
from datasets import Dataset

DATA_DIR = Path("/content/data")
DATA_DIR.mkdir(exist_ok=True)


def load_rows(path):
    if not path.exists():
        return None
    return [json.loads(line) for line in path.open(encoding="utf-8")]


train_rows = load_rows(DATA_DIR / "ecql_train.jsonl")
test_rows = load_rows(DATA_DIR / "ecql_test.jsonl")
if train_rows is None or test_rows is None:
    raise SystemExit(
        "Файлы ecql_train.jsonl / ecql_test.jsonl не найдены в /content/data. "
        "Загрузите их из локального репозитория и перезапустите ноутбук."
    )
print(f"Train: {len(train_rows)} | Test: {len(test_rows)}")


def build_sft_dataset(rows, tokenizer):
    """Токенизация пар с маскированием промпта: учимся только на ответе-ECQL."""

    def _tokenize(row):
        prefix = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": (
                    "Ты переводишь запросы с русского языка на корпоративный язык запросов ECQL. "
                    "Отвечай только самим ECQL-запросом, без пояснений.")},
                {"role": "user", "content": f"{row['instruction']}\n\n{row['input']}"},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        answer = tokenizer.encode(f"\n{row['output']}", add_special_tokens=False)
        return {
            "input_ids": prefix + answer + [tokenizer.eos_token_id],
            "attention_mask": [1] * (len(prefix) + len(answer) + 1),
            "labels": [-100] * len(prefix) + answer + [tokenizer.eos_token_id],
        }

    return Dataset.from_list([_tokenize(r) for r in rows])

# %% [markdown]
# ## 3. Загрузка модели (4-битный QLoRA) и токенизатора

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    TCFG["base_model"], quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(TCFG["base_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Модель загружена за {time.time() - t0:.1f} c")

# %% [markdown]
# ## 4. LoRA + SFTTrainer (trl)

# %%
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

lora_config = LoraConfig(
    r=TCFG["lora_r"],
    lora_alpha=TCFG["lora_alpha"],
    lora_dropout=TCFG["lora_dropout"],
    target_modules=TCFG["target_modules"],
    task_type="CAUSAL_LM",
    bias="none",
)

sft_config = SFTConfig(
    output_dir="/content/checkpoints",
    max_seq_length=TCFG["max_seq_length"],
    per_device_train_batch_size=TCFG["per_device_train_batch_size"],
    gradient_accumulation_steps=TCFG["gradient_accumulation_steps"],
    num_train_epochs=TCFG["num_train_epochs"],
    learning_rate=TCFG["learning_rate"],
    lr_scheduler_type=TCFG["lr_scheduler_type"],
    warmup_ratio=TCFG["warmup_ratio"],
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch",
    optim=TCFG["optim"],
    bf16=True,
    seed=42,
    report_to=[],
)

train_dataset = build_sft_dataset(train_rows, tokenizer)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)
print("Trainer готов, запускаем обучение...")

# %% [markdown]
# ## 5. Обучение (замер времени)

# %%
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"Обучение заняло: {elapsed:.1f} c = {elapsed / 60:.1f} мин")
print(f"Шагов: {result.global_step} | train_loss: {result.training_loss:.4f}")

# %% [markdown]
# ## 6. Сохранение адаптера

# %%
adapter_dir = Path("/content/ecql_adapter")
trainer.save_model(adapter_dir)
print(f"Адаптер сохранён: {adapter_dir}")

# Загрузка на HuggingFace Hub (раскомментируйте, вставив свой токен):
# trainer.push_to_hub("user/ecql-qwen3-4b-lora")

# %% [markdown]
# ## 7. Оценка на тестовой выборке (метрики + замер времени инференса)

# %%
def generate_ecql(model, tokenizer, instruction, user_input, max_new_tokens=128):
    messages = [
        {"role": "system", "content": (
            "Ты переводишь запросы с русского языка на корпоративный язык запросов ECQL. "
            "Отвечай только самим ECQL-запросом, без пояснений.")},
        {"role": "user", "content": f"{instruction}\n\n{user_input}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    prompt = prompt.to(model.device)
    with torch.no_grad():
        out = model.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt.shape[1]:], skip_special_tokens=True).strip()


model.eval()
preds = []
t0 = time.time()
for row in test_rows:
    preds.append(generate_ecql(model, tokenizer, row["instruction"], row["input"], MAX_NEW_TOKENS))
inf_elapsed = time.time() - t0
print(f"Инференс {len(test_rows)} примеров: {inf_elapsed:.1f} c "
      f"(в среднем {inf_elapsed / len(test_rows):.2f} c/пример)")

# %% [markdown]
# ## 8. Метрики (синтаксис через простой парсер ECQL)

# %%
import re

RE_QUERY = re.compile(
    r"^\s*FETCH\s+\[[A-Z_]+](?:\s+WHERE\s+.+?)?(?:\s+AS\s+(?:JSON|TABLE|LIST))?\s*$"
)


def syntax_ok(q):
    return bool(RE_QUERY.match(q))


valid = [p for p, r in zip(preds, test_rows) if syntax_ok(p)]
exact = [p for p, r in zip(preds, test_rows) if p.strip() == r["output"].strip()]
sql_leak = [p for p in preds if "SELECT" in p.upper() or re.search(r"\bAND\b|\bOR\b|>|<", p)]
print(f"Синтаксис корректный: {len(valid)}/{len(test_rows)} = {len(valid)/len(test_rows):.2%}")
print(f"Точное совпадение:    {len(exact)}/{len(test_rows)} = {len(exact)/len(test_rows):.2%}")
print(f"SQL-галлюцинации:     {len(sql_leak)}")

# %% [markdown]
# ## 9. Таблица вход — эталон — ответ

# %%
print(f"{'Вход (рус.)':<40}{'Эталон':<52}{'Ответ модели'}")
print("-" * 150)
for r, p in list(zip(test_rows, preds))[:5]:
    print(f"{r['input'][:39]:<40}{r['output'][:51]:<52}{p[:60]}")

# %% [markdown]
# ## 10. Итог
#
# - Время обучения и инференса замерены в ячейках 5 и 7 — сравните с CPU-версией
#   (`notebooks/report_local.py`, Qwen2.5-1.5B, LoRA r=4): на T4 ожидается ускорение
#   в ~5-10 раз и заметно более высокая точность синтаксиса за счёт более сильной модели.
# - Для полного отчёта с LLM-as-a-judge скопируйте `results/eval_local.json`
#   из локального проекта (или запустите `scripts/evaluate.py` локально).