"""Оценка обученного адаптера на тестовой выборке: метрики + LLM-as-a-judge.

Запуск:
    uv run python scripts/evaluate.py [--skip-judge]
"""

from __future__ import annotations

import argparse

from ecql.config import ADAPTER_DIR, DATA_DIR, load_yaml
from ecql.dataset import load_jsonl
from ecql.eval import evaluate_on_testset, print_eval_summary
from ecql.train import load_adapter_and_tokenizer
from ecql.yandex_client import YandexGPTClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true", help="Пропустить LLM-as-a-judge")
    args = parser.parse_args()

    cfg = load_yaml()
    test_rows = load_jsonl(DATA_DIR / "ecql_test.jsonl")
    print(f"Тестовых примеров: {len(test_rows)}")

    model, tokenizer = load_adapter_and_tokenizer(
        cfg["local_train"]["base_model"], ADAPTER_DIR, device="cpu"
    )
    model.eval()

    judge = None
    if not args.skip_judge:
        judge = YandexGPTClient(
            model=cfg["judge"]["model"],
            temperature=cfg["judge"]["temperature"],
            max_tokens=cfg["judge"]["max_tokens"],
        )

    result = evaluate_on_testset(
        model,
        tokenizer,
        test_rows,
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        device="cpu",
        judge=judge,
        use_judge=judge is not None,
    )
    print_eval_summary(result)


if __name__ == "__main__":
    main()
