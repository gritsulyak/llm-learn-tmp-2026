"""Оценка RAG-пайплайна метриками RAGAS.

Метрики:
- faithfulness          — насколько ответ опирается на контекст (без галлюцинаций)
- context_precision     — релевантность извлечённого контекста (LLMContextPrecisionWithReference)
- context_recall        — полнота контекста относительно эталонного ответа

Судья: Yandex Cloud (`yandexgpt-lite`, `yandexgpt-5.1`, ...) или Ollama (`qwen2.5:3b`) — выбирается флагом.

Запуск:
  uv run python scripts/evaluate_ragas.py --judge yandex                          # yandexgpt-lite
  uv run python scripts/evaluate_ragas.py --judge yandex --judge-model yandexgpt-5.1
  uv run python scripts/evaluate_ragas.py --judge ollama                           # qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
)
from ragas.run_config import RunConfig

from corporate_assistant.assistant import CorporateAssistant
from corporate_assistant.config import (
    COLLECTION_NAME,
    LLM_TEMPERATURE,
    OLLAMA_MODEL,
    OLLAMA_URL,
    TOP_K,
    YC_API_KEY,
    YC_FOLDER_ID,
    YC_MODEL,
    YC_URL,
)


# ---------------------------------------------------------------------------
# Langchain-совместимая обёртка над Yandex Cloud Foundation Models API
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
}


class ChatYandexGPT(BaseChatModel):
    """Минимальный langchain ChatModel для Yandex Cloud Foundation Models."""

    api_key: str = ""
    folder_id: str = ""
    model: str = "yandexgpt-lite"
    url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    temperature: float = 0.0
    max_tokens: int = 2000

    @property
    def _llm_type(self) -> str:
        return "yandexgpt"

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
        return [
            {"role": _ROLE_MAP.get(m.type, "user"), "text": m.content}
            for m in messages
        ]

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens),
            },
            "messages": messages,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}",
        }
        resp = requests.post(self.url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["result"]["alternatives"][0]["message"]["text"]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        api_messages = self._convert_messages(messages)
        text = self._call_api(api_messages)
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

# (вопрос, эталонный ответ)
GOLDEN = [
    (
        "За сколько дней нужно писать заявление на отпуск?",
        "Заявление на отпуск подается не позднее чем за 14 календарных дней до планируемой даты начала отпуска.",
    ),
    (
        "Какой размер суточных выплачивается при командировке и каков лимит расходов на проживание?",
        "Суточные — 1000 рублей за каждый календарный день. Проживание не более 6000 рублей за ночь "
        "для Москвы и Санкт-Петербурга и не более 4000 рублей за ночь для остальных городов.",
    ),
    (
        "Я уезжаю в командировку, а сразу после неё планирую уйти в отпуск. "
        "Какие сроки подачи заявлений нужно соблюсти для каждого события?",
        "На командировку — не позднее чем за 5 рабочих дней до выезда. На отпуск — не позднее чем за 14 "
        "календарных дней до начала отпуска.",
    ),
    (
        "Какие требования к корпоративным паролям и как часто их нужно менять?",
        "Пароль не менее 12 символов с буквами верхнего и нижнего регистра, цифрами и спецсимволами. "
        "Смена не реже одного раза в 90 дней.",
    ),
    (
        "Сотруднику со стажем 7 лет открыли больничный. Сколько процентов от заработка он получит?",
        "При стаже от 5 до 8 лет пособие составляет 80% среднего заработка.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS-оценка RAG-пайплайна")
    p.add_argument(
        "--judge",
        choices=["ollama", "yandex"],
        default="ollama",
        help="Модель-судья для RAGAS-метрик (по умолчанию: ollama)",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Имя модели судьи (для yandex: yandexgpt-lite, yandexgpt-5.1 и т.д.)",
    )
    return p.parse_args()


def _build_judge_llm(judge: str, judge_model: str | None = None) -> LangchainLLMWrapper:
    if judge == "yandex":
        return LangchainLLMWrapper(
            ChatYandexGPT(
                api_key=YC_API_KEY,
                folder_id=YC_FOLDER_ID,
                model=judge_model or YC_MODEL,
                url=YC_URL,
                temperature=LLM_TEMPERATURE,
            )
        )
    from langchain_ollama import ChatOllama

    return LangchainLLMWrapper(
        ChatOllama(model=judge_model or OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0, request_timeout=1800.0)
    )


def _build_embeddings(judge: str) -> LangchainEmbeddingsWrapper:
    from langchain_ollama import OllamaEmbeddings

    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="qwen2.5:3b", base_url=OLLAMA_URL)
    )


def main() -> None:
    args = parse_args()
    judge = args.judge
    judge_model = args.judge_model
    model_label = judge_model or ("qwen2.5:3b" if judge == "ollama" else YC_MODEL)
    print(f"Судья: {judge} ({model_label})")

    assistant = CorporateAssistant(collection_name=COLLECTION_NAME, top_k=TOP_K)

    samples: list[SingleTurnSample] = []
    for query, reference in GOLDEN:
        result = assistant.answer(query)
        contexts = [s["text"] for s in result.sources]
        # Цитаты [Файл.pdf, стр. N] — это ссылки на источник, а не факт из контекста;
        # судья RAGAS трактует их как «неподтверждённые утверждения» и занижает
        # faithfulness. Убираем их до оценки.
        response_no_citations = re.sub(r"\[[^\]]*\]", "", result.answer).strip()
        print(f"Q: {query}")
        print(f"  A: {result.answer[:120]}...")
        print(f"  dropped: {result.dropped_citations or 'нет'}")
        samples.append(
            SingleTurnSample(
                user_input=query,
                retrieved_contexts=contexts,
                response=response_no_citations,
                reference=reference,
            )
        )

    llm = _build_judge_llm(judge, judge_model)
    emb = _build_embeddings(judge)

    metrics = [
        Faithfulness(llm=llm),
        LLMContextPrecisionWithReference(llm=llm),
        ContextRecall(llm=llm),
    ]

    dataset = EvaluationDataset(samples=samples)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(timeout=1800, max_retries=3, max_wait=300),
    )

    print("\n=== RAGAS-результаты ===")
    print(score.to_pandas())
    print(score)

    slug = model_label.replace(":", "_").replace("/", "_")
    out = ROOT / "results" / f"ragas_scores_{judge}_{slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score._scores_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()