"""Валидация датасета: парсинг ECQL, дубликаты, покрытие сущностей/операторов.

Запуск:
    uv run python scripts/validate_dataset.py
"""

from __future__ import annotations

from ecql.config import DATASET_FULL, load_yaml, save_json
from ecql.dataset import dataset_stats, load_jsonl
from ecql.ecql_spec import ENTITIES, OPERATORS, parse_ecql


def main() -> None:
    rows = load_jsonl(DATASET_FULL)
    if not rows:
        print("Датасет пуст. Сначала запустите scripts/generate_dataset.py")
        return

    invalid = [r for r in rows if not parse_ecql(r["output"]).valid]
    dup_inputs = len(rows) - len({r["input"] for r in rows})
    min_len = min(len(r["input"]) for r in rows)
    max_len = max(len(r["input"]) for r in rows)

    stats = dataset_stats(rows)
    cfg = load_yaml()
    needed = d if (d := cfg.get("dataset", {})).get("min_pairs", 150) <= len(rows) else None
    ok = (not invalid) and dup_inputs == 0 and needed is not None

    print(f"Всего пар: {len(rows)}")
    print(f"Невалидных ECQL: {len(invalid)}")
    print(f"Дубликатов по input: {dup_inputs}")
    print(f"Длина русского запроса: {min_len}..{max_len} символов")
    print(f"Покрытие сущностей: {sorted(ENTITIES)} -> {sorted(stats['entities'])}")
    print(f"Покрытие операторов: {sorted(OPERATORS)} -> {sorted(stats['operators'])}")
    print(f"Требование min_pairs (150): {'выполнено' if needed else 'НЕ выполнено'}")
    print("ВЕРДИКТ:", "OK" if ok else "FAIL")

    validation = {
        "ok": ok,
        "n": len(rows),
        "invalid": len(invalid),
        "duplicates": dup_inputs,
    }
    save_json(Path("results") / "dataset_validation.json", validation)


if __name__ == "__main__":
    from pathlib import Path

    main()
