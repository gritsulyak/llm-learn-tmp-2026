"""Датасет ECQL: генерация моделью-учителем, валидация, разбиение, чтение/запись JSONL."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ecql.config import (
    DATASET_FULL,
    DATASET_TEST,
    DATASET_TRAIN,
    load_yaml,
)
from ecql.ecql_spec import parse_ecql
from ecql.yandex_client import YandexGPTClient, parse_teacher_rows

TEACHER_SYSTEM = (
    "Ты — модель-учитель, которая генерирует обучающий датасет для перевода "
    "русских запросов менеджеров в корпоративный язык запросов ECQL компании E-Corp."
)

_SPEC_TEXT = """
Спецификация языка ECQL:
1. Запрос всегда начинается с ключевого слова FETCH.
2. Сущности — заглавными буквами в квадратных скобках: [EMPLOYEES], [PROJECTS], [INVENTORY], [DEALS].
3. Поля сущностей:
   [EMPLOYEES]: @name, @city, @department, @position, @salary, @age, @status, @experience
   [PROJECTS]: @name, @status, @budget, @deadline, @owner, @team_size, @priority
   [INVENTORY]: @name, @category, @quantity, @price, @location, @status
   [DEALS]: @client, @amount, @currency, @date, @status, @manager
4. Операторы сравнения: IS (равно), NOT (не равно), ABOVE (больше), BELOW (меньше).
5. Строковые значения — в одинарных кавычках: @city IS 'Moscow'.
6. Числовые значения — без кавычек: @salary ABOVE 150000.
7. Логические связки: && (И) и || (ИЛИ). Сложные выражения можно группировать скобками.
8. Формат вывода (необязательно, в конце запроса): AS JSON, AS TABLE или AS LIST.
"""


def teacher_prompt(count: int) -> str:
    return (
        _SPEC_TEXT
        + f"\nСгенерируй {count} разнообразных пар «запрос на русском языке» -> «запрос ECQL».\n"
        "Требования к разнообразию:\n"
        "- от простых (одна сущность, одно условие) до сложных (2-3 условия, && и ||, скобки, AS-формат);\n"
        "- покрыть все 4 сущности, разные поля и все операторы IS/NOT/ABOVE/BELOW;\n"
        "- разные города, должности, статусы, суммы, валюты (USD, EUR, RUB);\n"
        "- русский запрос формулируй как вопрос менеджера, а не как перевод DSL;\n"
        "- используй и точные числа, и строковые значения.\n"
        "Верни строго JSON Lines (по одному объекту на строку) без markdown и пояснений:\n"
        '{"instruction": "Переведи запрос на ECQL", "input": "<русский запрос>", "output": "<ECQL-запрос>"}'
    )


def generate_dataset(
    client: YandexGPTClient,
    target_size: int = 200,
    per_batch: int = 20,
    max_calls: int = 20,
) -> list[dict[str, str]]:
    """Генерирует датасет моделью-учителем, валидирует и дедуплицирует."""
    rows: list[dict[str, str]] = []
    seen_inputs: set[str] = set()
    calls = 0
    while len(rows) < target_size and calls < max_calls:
        calls += 1
        text = client.generate_examples(teacher_prompt(per_batch), system=TEACHER_SYSTEM)
        for row in parse_teacher_rows(text):
            inp = row["input"]
            out = row["output"]
            if inp in seen_inputs:
                continue
            if not parse_ecql(out).valid:
                continue
            seen_inputs.add(inp)
            rows.append(row)
        print(f"  [вызов {calls}] накоплено: {len(rows)}/{target_size}")
        if len(rows) < target_size and calls >= max_calls:
            break
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_dataset(
    rows: list[dict], train_ratio: float = 0.8, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


def build_dataset_files(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Сохраняет полный датасет и train/test split в data/."""
    cfg = load_yaml()
    d = cfg.get("dataset", {})
    save_jsonl(rows, DATASET_FULL)
    train, test = split_dataset(rows, d.get("train_ratio", 0.8), d.get("seed", 42))
    save_jsonl(train, DATASET_TRAIN)
    save_jsonl(test, DATASET_TEST)
    return train, test


def dataset_stats(rows: list[dict]) -> dict:
    entities: dict[str, int] = {}
    ops: dict[str, int] = {}
    formats: dict[str, int] = {}
    multi_cond = 0
    for row in rows:
        p = parse_ecql(row["output"])
        entities[p.entity] = entities.get(p.entity, 0) + 1
        for c in p.conditions:
            ops[c["op"]] = ops.get(c["op"], 0) + 1
        if p.output_format:
            formats[p.output_format] = formats.get(p.output_format, 0) + 1
        if len(p.conditions) > 1:
            multi_cond += 1
    return {
        "total": len(rows),
        "entities": dict(sorted(entities.items())),
        "operators": dict(sorted(ops.items())),
        "output_formats": dict(sorted(formats.items())),
        "multi_condition": multi_cond,
    }


def build_sft_text(instruction: str, user_input: str, tokenizer, system: str | None = None) -> str:
    """Собирает промпт для SFT по chat-шаблону Qwen2.5."""
    system_prompt = system or (
        "Ты переводишь запросы с русского языка на корпоративный язык запросов ECQL. "
        "Отвечай только самим ECQL-запросом, без пояснений."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}\n\n{user_input}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
