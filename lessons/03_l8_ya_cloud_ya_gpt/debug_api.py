"""
Запусти: uv run debug_api.py
Покажет точный ответ API и где падает индексация
"""
import os, json, requests
from dotenv import load_dotenv

load_dotenv()

YC_API_KEY   = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")

print(f"YC_FOLDER_ID: {'OK (' + YC_FOLDER_ID[:4] + '...)' if YC_FOLDER_ID else 'НЕ ЗАДАН!'}")
print(f"YC_API_KEY:   {'OK (' + YC_API_KEY[:4] + '...)' if YC_API_KEY else 'НЕ ЗАДАН!'}")

payload = {
    "modelUri": f"gpt://{YC_FOLDER_ID}/yandexgpt-lite",
    "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": "100"},
    "messages": [
        {"role": "system", "text": "Отвечай кратко."},
        {"role": "user",   "text": "Скажи: OK"},
    ]
}
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Api-Key {YC_API_KEY}"
}

print("\n--- Отправляю запрос ---")
resp = requests.post(
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
    headers=headers, json=payload, timeout=30
)

print(f"HTTP статус: {resp.status_code}")
print(f"\n--- Сырой JSON ответа ---")
data = resp.json()
print(json.dumps(data, ensure_ascii=False, indent=2))

print(f"\n--- Навигация по ответу ---")
result = data.get("result")
print(f"data['result'] тип: {type(result)}")

if isinstance(result, dict):
    alts = result.get("alternatives")
    print(f"result['alternatives'] тип: {type(alts)}, значение: {alts}")
    if isinstance(alts, list) and len(alts) > 0:
        print(f"alts[0]: {alts[0]}")
        print(f"\n✅ Правильный путь: data['result']['alternatives'][0]['message']['text']")
        print(f"Значение: {alts[0]['message']['text']}")
    else:
        print(f"❌ alternatives не список или пустой: {alts}")
elif isinstance(result, list):
    print(f"❌ result — список, а не словарь! Длина: {len(result)}")
    print(f"result[0]: {result[0] if result else 'пусто'}")
    print(f"\n✅ Правильный путь будет: data['result'][0]['alternatives'][0]['message']['text']")
else:
    print(f"❌ result = {result}")
    if data.get("error"):
        print(f"Ошибка API: {data['error']}")
