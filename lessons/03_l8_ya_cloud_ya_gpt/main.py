import os

from dotenv import load_dotenv
import requests

load_dotenv()

YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")

# to debug yandex api
prompt = {
    "modelUri": f"gpt://{YC_FOLDER_ID}/yandexgpt-lite",
    "completionOptions": {
        "stream": False,
        "temperature": 0.1,
        "maxTokens": "1000"
    },
    "messages": [
        {
            "role": "system",
            "text": "Ты ассистент дроид, способный помочь в галактических приключениях."
        },
        {
            "role": "user",
            "text": "Привет, Дроид! Мне нужна твоя помощь, чтобы узнать больше о Силе. Как я могу научиться ее использовать?"
        },
        {
            "role": "assistant",
            "text": "Привет! Чтобы овладеть Силой, тебе нужно понять ее природу. Сила находится вокруг нас и соединяет всю галактику. Начнем с основ медитации."
        },
        {
            "role": "user",
            "text": "Хорошо, а как насчет строения светового меча? Это важная часть тренировки джедая. Как мне создать его?"
        }
    ]
}


url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Api-Key {YC_API_KEY}"
}


response = requests.post(url, headers=headers, json=prompt)

with open("result.json", "w", encoding="utf-8") as f:
    f.write(response.text)
print(
    response.json()
    .get("result")
    .get("alternatives")[0]
    .get("message")
    .get("text")
)