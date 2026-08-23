"""LoRA-дообучение (SFT) малой LLM на CPU.

Пайплайн:
1. токенизация пар «русский запрос -> ECQL» через chat-шаблон с маскированием
   промпта (labels = -100 для токенов пользователя/системы);
2. PEFT LoRA (r=4, alpha=8) на q/k/v/o проекциях;
3. обучение через `trl.SFTTrainer` (transformers Trainer под капотом);
4. логирование Loss/времени в CSV и сохранение адаптера.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer

from ecql.config import LOSS_LOG


def build_sft_dataset(rows: list[dict], tokenizer) -> Dataset:
    """Токенизирует пары и маскирует промпт (обучаем только на ответе-ECQL).

    Qwen2.5 не поддерживает `return_assistant_tokens_mask` в chat-шаблоне,
    поэтому границу промпта определяем через `add_generation_prompt=True`:
    префикс заканчивается токеном начала ответа ассистента.
    """

    def _tokenize(row: dict) -> dict:
        system_prompt = (
            "Ты переводишь запросы с русского языка на корпоративный язык запросов ECQL. "
            "Отвечай только самим ECQL-запросом, без пояснений."
        )
        prefix_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{row['instruction']}\n\n{row['input']}"},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        answer_ids = tokenizer.encode(f"\n{row['output']}", add_special_tokens=False)
        input_ids = prefix_ids + answer_ids + [tokenizer.eos_token_id]
        labels = (
            [-100] * len(prefix_ids)
            + answer_ids
            + [tokenizer.eos_token_id]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return Dataset.from_list([_tokenize(r) for r in rows])


class LossLoggingCallback(TrainerCallback):
    """Пишет (step, loss, elapsed_sec, lr) в CSV в реальном времени."""

    def __init__(self, path: Path, start_time: float) -> None:
        self.path = path
        self.start_time = start_time
        self._fh = None

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        if "loss" not in logs:
            return
        self._fh = self._fh or self.path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(self._fh)
        if state.global_step == 0 or self.path.stat().st_size == 0:
            writer.writerow(["step", "loss", "elapsed_sec", "lr"])
        writer.writerow(
            [
                state.global_step,
                round(float(logs["loss"]), 5),
                round(time.time() - self.start_time, 1),
                logs.get("learning_rate", ""),
            ]
        )
        self._fh.flush()

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self._fh:
            self._fh.close()


def apply_lora(model, cfg: dict, device: str):
    """Навешивает LoRA-адаптер на модель (r/alpha из конфига)."""
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    if device == "cuda":
        return model, lora  # SFTTrainer применит peft_config при QLoRA
    peft_model = get_peft_model(model, lora)
    peft_model.print_trainable_parameters()
    return peft_model, None


def make_trainer(
    rows: list[dict],
    cfg: dict,
    output_dir: Path,
    loss_log: Path,
    start_time: float,
):
    """Собирает SFTTrainer для CPU (fp32) или GPU (4-bit QLoRA)."""
    device = cfg["device"]
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    load_kwargs = {}
    if device == "cuda" and cfg.get("load_in_4bit"):
        load_kwargs = {
            "load_in_4bit": True,
            "quantization_config": None,
            "device_map": "auto",
        }
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype,
        **load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_sft_dataset(rows, tokenizer)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_seq_length=cfg["max_seq_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_strategy="steps",
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg.get("save_strategy", "epoch"),
        optim=cfg["optim"],
        bf16=device == "cuda",
        fp16=False,
        disable_tqdm=False,
        seed=42,
        report_to=[],
    )

    model, peft_config = apply_lora(model, cfg, device)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[LossLoggingCallback(loss_log, start_time)],
    )
    return trainer, tokenizer


def train(
    rows: list[dict],
    cfg: dict,
    output_dir: Path,
    loss_log: Path = LOSS_LOG,
    start_time: float | None = None,
) -> dict:
    """Запускает SFT-обучение и возвращает сводку (время, число шагов, финальный loss)."""
    start_time = start_time or time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_log.parent.mkdir(parents=True, exist_ok=True)

    trainer, tokenizer = make_trainer(rows, cfg, output_dir, loss_log, start_time)
    result = trainer.train()
    trainer.save_model(output_dir)

    try:
        final_loss = float(result.training_loss)
    except (TypeError, AttributeError):
        final_loss = None

    return {
        "elapsed_sec": round(time.time() - start_time, 1),
        "global_step": result.global_step,
        "training_loss": final_loss,
        "train_samples": len(rows),
        "base_model": cfg["base_model"],
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "device": cfg["device"],
        "adapter_dir": str(output_dir),
    }


def load_adapter_and_tokenizer(base_model: str, adapter_dir: Path, device: str = "cpu"):
    """Загружает базовую модель + LoRA-адаптер для инференса."""
    from peft import PeftModel

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
