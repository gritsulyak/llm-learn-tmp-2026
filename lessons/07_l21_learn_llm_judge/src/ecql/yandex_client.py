"""Клиент Yandex Cloud Foundation Models: модель-учитель (генерация датасета) и судья (LLM-as-a-judge)."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from ecql.config import YC_API_KEY, YC_FOLDER_ID, YC_MODEL, YC_URL


class YandexGPTClient:
    """Тонкая обёртка над API YandexGPT (Foundation Models, gpt://.../completion)."""

    def __init__(
        self,
        model: str = YC_MODEL,
        api_key: str = YC_API_KEY,
        folder_id: str = YC_FOLDER_ID,
        url: str = YC_URL,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.folder_id = folder_id
        self.url = url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "text": system})
        messages.append({"role": "user", "text": prompt})
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature if temperature is None else temperature,
                "maxTokens": str(self.max_tokens if max_tokens is None else max_tokens),
            },
            "messages": messages,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
        }
        resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["result"]["alternatives"][0]["message"]["text"]

    def generate_examples(self, prompt: str, system: str | None = None) -> str:
        return self.complete(prompt, system=system)


def extract_json_lines(text: str) -> list[dict]:
    """Извлекает объекты JSON из ответа учителя (устойчив к markdown-обёртке)."""
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    objects: list[dict] = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        if text[idx] == "{":
            try:
                obj, end = decoder.raw_decode(text, idx)
                objects.append(obj)
                idx = end
                continue
            except json.JSONDecodeError:
                pass
        idx += 1
    return objects


def parse_teacher_rows(text: str) -> list[dict[str, str]]:
    """Нормализует ответ учителя в список пар instruction/input/output."""
    rows: list[dict[str, str]] = []
    for obj in extract_json_lines(text):
        if not isinstance(obj, dict):
            continue
        out = str(obj.get("output", "")).strip()
        inp = str(obj.get("input", "")).strip()
        instr = str(obj.get("instruction", "Переведи запрос на ECQL")).strip()
        if inp and out:
            rows.append({"instruction": instr, "input": inp, "output": out})
    return rows


def judge_ecql_pair(client: YandexGPTClient, row: dict, prediction: str) -> dict[str, Any]:
    """LLM-as-a-judge: сравнивает предсказание модели с эталоном ECQL.

    Возвращает score (0-5), verdict (correct/partial/incorrect) и пояснение.
    """
    prompt = (
        "Ты — строгий эксперт-судья по корпоративному языку запросов ECQL. "
        "Оцени ответ дообученной модели по трём критериям:\n"
        "1) синтаксис (верно ли расставлены FETCH, [], @, &&/||, AS);\n"
        "2) логическая точность (та ли сущность, поля и операторы IS/NOT/ABOVE/BELOW);\n"
        "3) отсутствие галлюцинаций (не подменил ли ответ SQL-подобным синтаксисом).\n\n"
        f"Запрос на русском: {row['input']}\n"
        f"Эталонный ECQL: {row['output']}\n"
        f"Ответ модели: {prediction}\n\n"
        "Ответь строго одним JSON-объектом без пояснений:\n"
        '{"score": <целое от 0 до 5>, "verdict": "correct"|"partial"|"incorrect", '
        '"comment": "<краткое пояснение на русском>"}'
    )
    text = client.complete(
        prompt,
        temperature=0.0,
        max_tokens=800,
        system="Ты возвращаешь только валидный JSON.",
    )
    return parse_judge(text, row["input"], prediction)


def parse_judge(text: str, user_input: str = "", prediction: str = "") -> dict:
    """Разбирает JSON-ответ судьи с защитой от шума вокруг."""
    objs = extract_json_lines(text)
    if not objs:
        return {
            "score": 0,
            "verdict": "error",
            "comment": "Судья не вернул JSON",
            "raw_judge": text[:500],
            "input": user_input,
            "prediction": prediction,
        }
    obj = objs[0]
    return {
        "score": int(obj.get("score", 0)),
        "verdict": str(obj.get("verdict", "incorrect")),
        "comment": str(obj.get("comment", "")),
        "raw_judge": text[:500],
        "input": user_input,
        "prediction": prediction,
    }
