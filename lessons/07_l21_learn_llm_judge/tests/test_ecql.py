"""Тесты: ECQL-грамматика, валидатор, датасет, судья."""

from __future__ import annotations

from ecql.dataset import split_dataset, teacher_prompt
from ecql.ecql_spec import (
    _split_conditions,
    describe,
    is_sql_style,
    parse_ecql,
)
from ecql.yandex_client import extract_json_lines, parse_judge, parse_teacher_rows


def test_parse_simple():
    p = parse_ecql("FETCH [EMPLOYEES] WHERE @city IS 'Moscow' && @salary ABOVE 150000")
    assert p.valid
    assert p.entity == "EMPLOYEES"
    assert len(p.conditions) == 2
    assert p.operators == ["&&"]


def test_parse_nested_parens():
    q = "FETCH [DEALS] WHERE @client IS 'Acme' || (@amount ABOVE 500000 && @currency IS 'USD') AS JSON"
    p = parse_ecql(q)
    assert p.valid
    assert len(p.conditions) == 3
    assert p.output_format == "JSON"


def test_parse_invalid_sql():
    assert not parse_ecql("SELECT * FROM employees WHERE city = 'Moscow'").valid
    assert is_sql_style("SELECT * FROM employees WHERE city = 'Moscow'")


def test_parse_unknown_entity():
    p = parse_ecql("FETCH [FOO] WHERE @city IS 'X'")
    assert not p.valid
    assert any("сущности" in e for e in p.errors)


def test_parse_wrong_field():
    p = parse_ecql("FETCH [EMPLOYEES] WHERE @budget IS 'X'")
    assert not p.valid
    assert any("@budget" in e for e in p.errors)


def test_parse_above_requires_number():
    assert not parse_ecql("FETCH [EMPLOYEES] WHERE @salary ABOVE 'high'").valid


def test_parse_no_where():
    p = parse_ecql("FETCH [INVENTORY] AS LIST")
    assert p.valid
    assert p.output_format == "LIST"
    assert p.conditions == []


def test_split_conditions_respects_parens():
    parts, ops = _split_conditions("(@a IS 'x' && @b ABOVE 1) || @c NOT 'y'")
    assert len(parts) == 2
    assert ops == ["||"]


def test_describe():
    d = describe("FETCH [EMPLOYEES] WHERE @salary ABOVE 100 && @city IS 'SPb'")
    assert d["entity"] == "EMPLOYEES"
    assert d["fields"] == ["city", "salary"]
    assert "ABOVE" in d["ops"] and "IS" in d["ops"]


def test_split_dataset_ratio():
    rows = [{"input": str(i), "output": f"FETCH [EMPLOYEES] WHERE @age IS {i}"} for i in range(100)]
    train, test = split_dataset(rows, train_ratio=0.8, seed=42)
    assert len(train) == 80
    assert len(test) == 20
    assert {r["input"] for r in train} | {r["input"] for r in test} == {r["input"] for r in rows}


def test_teacher_prompt_contains_spec():
    prompt = teacher_prompt(5)
    assert "FETCH" in prompt
    assert "EMPLOYEES" in prompt
    assert "ABOVE" in prompt
    assert "JSON Lines" in prompt


def test_extract_json_lines_markdown():
    text = (
        "```json\n"
        '{"instruction": "Переведи запрос на ECQL", "input": "а", "output": "FETCH [DEALS] WHERE @a IS 1"}\n'
        "```"
    )
    objs = extract_json_lines(text)
    assert len(objs) == 1
    assert objs[0]["input"] == "а"


def test_parse_teacher_rows():
    text = (
        '{"instruction": "i", "input": "in", "output": "out"}\n'
        '{"input": "x", "output": "y"}\n'
        '{"input": "", "output": "z"}\n'
    )
    rows = parse_teacher_rows(text)
    assert len(rows) == 2
    assert rows[1]["instruction"] == "Переведи запрос на ECQL"


def test_parse_judge_valid():
    res = parse_judge(
        '{"score": 5, "verdict": "correct", "comment": "идеально"}',
        user_input="u",
        prediction="p",
    )
    assert res["score"] == 5
    assert res["verdict"] == "correct"


def test_parse_judge_error_fallback():
    res = parse_judge("no json here")
    assert res["verdict"] == "error"
    assert res["score"] == 0
