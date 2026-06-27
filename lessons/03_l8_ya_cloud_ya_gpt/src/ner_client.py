# src/ner_client.py
import os, json, requests
from dotenv import load_dotenv

load_dotenv()

YC_API_KEY   = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

SYSTEM_PROMPT = """Ты — точный AI-помощник юриста.
Извлекай факты строго из текста, без домыслов.
Если данных нет — верни null. Никогда не придумывай."""

USER_PROMPT_TEMPLATE = """Извлеки из текста договора об оказании услуг:
- parties: список сторон (name, inn, kpp или null)
- dates: дата подписания и сроки действия
- amounts: суммы с валютой или null
- obligations: сроки обязательств (дни/месяцы/даты) или null

Верни ТОЛЬКО валидный JSON без markdown и пояснений.

Пример:
{{
  "parties": [
    {{"name": "ООО Ромашка", "inn": "7701234567", "kpp": "770101001"}},
    {{"name": "ИП Иванов И.И.", "inn": "540312345678", "kpp": null}}
  ],
  "dates": {{"signed": "15.03.2024", "valid_from": "01.04.2024", "valid_until": "30.06.2024"}},
  "amounts": "350 000 рублей, в т.ч. НДС 20%",
  "obligations": "оплата в течение 10 рабочих дней с момента подписания акта"
}}

Текст договора:
{text}"""


def clean_json(raw: str) -> str:
    """Убирает markdown-обёртку ```json ... ``` если модель её добавила."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # убираем первую строку (```json или ```) и последнюю (```)
        lines = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _call_api(text: str) -> str:
    payload = {
        "modelUri": f"gpt://{YC_FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.1,
            "maxTokens": "1000"
        },
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user",   "text": USER_PROMPT_TEMPLATE.format(text=text)},
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YC_API_KEY}"
    }
    resp = requests.post(URL, headers=headers, json=payload, timeout=30)
    print(f" !!!  resp: {resp}")
    resp.raise_for_status()
    # alternatives — это список, нужен индекс [0]
    return resp.json()["result"]["alternatives"][0]["message"]["text"]


def extract_entities(text: str) -> dict:
    raw     = _call_api(text)
    print(f" !!!  raw in extract: {raw}")
    cleaned = clean_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # сохраняем сырой ответ для анализа ошибок в отчёте
        return {"error": str(e), "raw_response": raw}