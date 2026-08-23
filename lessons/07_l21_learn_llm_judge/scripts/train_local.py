"""Локальное дообучение на CPU: Qwen2.5-1.5B-Instruct + LoRA (r=4, alpha=8).

Запуск:
    uv run python scripts/train_local.py [--epochs 2] [--batch 1] [--grad-accum 4]
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from ecql.config import ADAPTER_DIR, DATA_DIR, LOSS_LOG, RESULTS_DIR, load_yaml, save_json
from ecql.dataset import load_jsonl
from ecql.train import train


def main() -> None:
    torch.set_num_threads(16)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml()
    tcfg = dict(cfg["local_train"])
    if args.epochs:
        tcfg["num_train_epochs"] = args.epochs
    if args.batch:
        tcfg["per_device_train_batch_size"] = args.batch
    if args.grad_accum:
        tcfg["gradient_accumulation_steps"] = args.grad_accum

    train_rows = load_jsonl(DATA_DIR / "ecql_train.jsonl")
    print(f"Обучающих примеров: {len(train_rows)}")
    print(f"Модель: {tcfg['base_model']} | LoRA r={tcfg['lora_r']} alpha={tcfg['lora_alpha']} "
          f"| device={tcfg['device']} | epochs={tcfg['num_train_epochs']}")

    start = time.time()
    summary = train(train_rows, tcfg, ADAPTER_DIR, LOSS_LOG, start_time=start)
    save_json(RESULTS_DIR / "train_local.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Адаптер сохранён: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
