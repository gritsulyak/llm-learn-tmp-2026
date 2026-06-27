import os, json, requests
from dotenv import load_dotenv

load_dotenv()

# Флаг переключения режимов (True/False)
IS_LOCAL = os.getenv("IS_LOCAL", "false").lower() in ("true", "1", "yes")

# Конфигурация Yandex Cloud
YC_API_KEY   = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
YC_URL       = os.getenv("YC_URL")

# Конфигурация Ollama
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL   = f"{OLLAMA_HOST}/v1/chat/completions"

SYSTEM_PROMPT = """Ты — точный AI-помощник юриста.
Извлекай факты только из текста, без домыслов.
Если данных нет — верни null. Никогда не придумывай."""

USER_PROMPT_TEMPLATE = """Извлеки из текста договора об оказании услуг:
- parties: список сторон (name, inn, kpp)
- dates: дата подписания и сроки действия
- amounts: суммы с валютой
- obligations: только сроки обязательств (дни/месяцы/даты) если есть

Верни ТОЛЬКО валидный JSON без markdown и пояснений.
Если поля нет — верни null.

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
        lines = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _call_api(text: str) -> str:
    # Формируем промпт пользователя
    user_content = USER_PROMPT_TEMPLATE.format(text=text)
    
    if IS_LOCAL:
        # Конфигурация для локальной Ollama
        url = OLLAMA_URL
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": OLLAMA_MODEL,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "response_format": { "type": "json_object" }
        }
        timeout = 300  # Локальные модели на CPU/GPU могут отвечать дольше
    else:
        # Конфигурация для Yandex Cloud
        url = YC_URL
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YC_API_KEY}"
        }
        payload = {
            "modelUri": f"gpt://{YC_FOLDER_ID}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.01,
                "maxTokens": "1000"
            },
            "messages": [
                {"role": "system", "text": SYSTEM_PROMPT},
                {"role": "user",   "text": user_content},
            ]
        }
        timeout = 30

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    print(f" !!!  resp: {resp}")
    resp.raise_for_status()
    
    # Извлечение текста в зависимости от структуры ответа API
    response_json = resp.json()
    if IS_LOCAL:
        return response_json["choices"][0]["message"]["content"]
    else:
        return response_json["result"]["alternatives"][0]["message"]["text"]


def extract_entities(text: str) -> dict:
    raw     = _call_api(text)
    print(f" !!!  raw in extract: {raw}")
    cleaned = clean_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw_response": raw}
