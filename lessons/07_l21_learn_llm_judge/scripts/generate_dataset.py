"""Генерация датасета ECQL моделью-учителем (YandexGPT Pro) + валидация + train/test split.

Запуск:
    uv run python scripts/generate_dataset.py [--target 200] [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ecql.config import DATASET_FULL, load_yaml, save_json
from ecql.dataset import (
    build_dataset_files,
    dataset_stats,
    generate_dataset,
    load_jsonl,
    save_jsonl,
)
from ecql.yandex_client import YandexGPTClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=None, help="Целевой размер датасета")
    parser.add_argument("--force", action="store_true", help="Перегенерировать, даже если датасет есть")
    args = parser.parse_args()

    cfg = load_yaml()
    d = cfg.get("dataset", {})
    target = args.target or d.get("target_size", 200)

    if DATASET_FULL.exists() and not args.force:
        rows = load_jsonl(DATASET_FULL)
        print(f"Датасет уже существует: {len(rows)} пар ({DATASET_FULL})")
        train, test = build_dataset_files(rows)
    else:
        client = YandexGPTClient(
            model=cfg["teacher"]["model"],
            temperature=cfg["teacher"]["temperature"],
            max_tokens=cfg["teacher"]["max_tokens"],
        )
        print(f"Генерация датасета моделью-учителем {cfg['teacher']['model']}...")
        rows = generate_dataset(
            client,
            target_size=target,
            per_batch=d.get("per_batch", 20),
        )
        print(f"Сгенерировано валидных пар: {len(rows)}")
        save_jsonl(rows, DATASET_FULL)
        train, test = build_dataset_files(rows)

    stats = dataset_stats(train + test)
    print("\nСтатистика датасета:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    save_json(Path("results") / "dataset_stats.json", stats)
    print(f"\nTrain: {len(train)} | Test: {len(test)}")
    print(f"Файлы: {DATASET_FULL}, {Path('data/ecql_train.jsonl')}, {Path('data/ecql_test.jsonl')}")


if __name__ == "__main__":
    main()
