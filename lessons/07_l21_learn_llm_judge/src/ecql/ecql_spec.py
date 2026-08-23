"""Спецификация языка ECQL компании E-Corp: грамматика, парсер, валидатор.

Грамматика (обобщённая):
    query    := "FETCH" "[" ENTITY "]" ( "WHERE" expr )? ( "AS" FORMAT )?
    expr     := clause ( ( "&&" | "||" ) clause )*
    clause   := "(" expr ")" | condition
    condition:= "@" FIELD OP VALUE
    OP       := "IS" | "NOT" | "ABOVE" | "BELOW"
    VALUE    := "'" any "'" | number
    FORMAT   := "JSON" | "TABLE" | "LIST"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ENTITIES: dict[str, set[str]] = {
    "EMPLOYEES": {
        "name", "city", "department", "position", "salary", "age", "status",
        "experience",
    },
    "PROJECTS": {
        "name", "status", "budget", "deadline", "owner", "team_size", "priority",
    },
    "INVENTORY": {
        "name", "category", "quantity", "price", "location", "status",
    },
    "DEALS": {
        "client", "amount", "currency", "date", "status", "manager",
    },
}

OPERATORS = {"IS", "NOT", "ABOVE", "BELOW"}
FORMATS = {"JSON", "TABLE", "LIST"}
LOGIC_AND = "&&"
LOGIC_OR = "||"

RE_QUERY = re.compile(
    r"^\s*FETCH\s+\[(?P<entity>[A-Z][A-Z_]*)\]"
    r"(?:\s+WHERE\s+(?P<where>.+?))?"
    r"(?:\s+AS\s+(?P<format>JSON|TABLE|LIST))?\s*$",
    re.DOTALL,
)
RE_COND = re.compile(
    r"^\s*@(?P<field>[a-z][a-z_]*)\s+"
    r"(?P<op>IS|NOT|ABOVE|BELOW)\s+"
    r"(?P<value>'(?:[^'\\]|\\.)*'|-?\d+(?:\.\d+)?)\s*$"
)


@dataclass
class ParsedQuery:
    entity: str = ""
    conditions: list[dict] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)  # логические связки между условиями
    output_format: str | None = None
    errors: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def valid(self) -> bool:
        return not self.errors


def _split_conditions(expr: str) -> tuple[list[str], list[str]]:
    """Разбивает WHERE-выражение на отдельные условия, учитывая скобки."""
    parts: list[str] = []
    ops: list[str] = []
    depth = 0
    current = ""
    i = 0
    n = len(expr)
    while i < n:
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
        if depth == 0 and expr.startswith(LOGIC_AND, i):
            parts.append(current)
            ops.append(LOGIC_AND)
            current = ""
            i += 2
            continue
        if depth == 0 and expr.startswith(LOGIC_OR, i):
            parts.append(current)
            ops.append(LOGIC_OR)
            current = ""
            i += 2
            continue
        current += expr[i]
        i += 1
    parts.append(current)
    return parts, ops


def _validate_condition(
    cond: str, entity: str, errors: list[str], collected: list[dict] | None = None
) -> bool:
    cond = cond.strip()
    if cond.startswith("(") and cond.endswith(")"):
        return _validate_where(cond[1:-1], entity, errors, collected)
    m = RE_COND.match(cond)
    if not m:
        errors.append(f"Некорректное условие: {cond!r}")
        return False
    field, op, value = m.group("field"), m.group("op"), m.group("value")
    if op not in OPERATORS:
        errors.append(f"Неизвестный оператор: {op}")
        return False
    if field not in ENTITIES.get(entity, set()):
        errors.append(f"Поле @{field} не существует у сущности [{entity}]")
        return False
    if op in {"ABOVE", "BELOW"}:
        if not re.match(r"^-?\d+(\.\d+)?$", value):
            errors.append(f"Оператор {op} требует числовое значение: {value!r}")
            return False
    if collected is not None:
        collected.append({"field": field, "op": op, "value": value})
    return True


def _validate_where(
    expr: str, entity: str, errors: list[str], collected: list[dict] | None = None
) -> bool:
    parts, ops = _split_conditions(expr)
    if not parts:
        errors.append("Пустое WHERE-выражение")
        return False
    if len(ops) + 1 != len(parts):
        errors.append("Некорректное сочетание логических связок")
        return False
    ok = True
    for part in parts:
        ok = _validate_condition(part, entity, errors, collected) and ok
    return ok


def parse_ecql(query: str) -> ParsedQuery:
    """Разбирает строку ECQL и возвращает структурированный результат.

    Парсер не пытается исправлять ошибки: любое несоответствие грамматике
    попадает в `errors`, а флаг `valid` становится False.
    """
    q = query.strip().strip("`")
    result = ParsedQuery(raw=q)
    m = RE_QUERY.match(q)
    if not m:
        result.errors.append("Запрос должен начинаться с FETCH [СУЩНОСТЬ]")
        return result

    entity = m.group("entity")
    result.entity = entity
    if entity not in ENTITIES:
        result.errors.append(f"Неизвестная сущность [{entity}]")

    where = m.group("where")
    if where:
        _validate_where(where, entity, result.errors, result.conditions)
        _, ops = _split_conditions(where)
        result.operators = ops

    fmt = m.group("format")
    if fmt and fmt not in FORMATS:
        result.errors.append(f"Неизвестный формат вывода: {fmt}")
    result.output_format = fmt

    return result


def strip_sql(query: str) -> str:
    """Возвращает признак SQL-галлюцинации: True, если в ответе есть SQL-конструкции."""
    return query


def is_sql_style(query: str) -> bool:
    """True, если модель попыталась выдать SQL вместо ECQL (галлюцинация)."""
    q = query.strip().upper()
    markers = (
        "SELECT ",
        " FROM ",
        " WHERE ",
        "JOIN",
        "GROUP BY",
        "ORDER BY",
        "CREATE TABLE",
        "INSERT INTO",
        "UPDATE ",
    )
    return any(m in q for m in markers) and not q.startswith("FETCH")


def describe(query: str) -> dict:
    """Компактное описание разобранного ECQL для сравнения с эталоном."""
    p = parse_ecql(query)
    return {
        "entity": p.entity if p.valid else None,
        "fields": sorted({c["field"] for c in p.conditions}),
        "ops": sorted({c["op"] for c in p.conditions}),
        "logic": p.operators,
        "format": p.output_format,
    }
