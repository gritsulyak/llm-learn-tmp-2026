"""Оценка RAG-пайплайна метриками RAGAS на локальной модели.

Метрики:
- faithfulness          — насколько ответ опирается на контекст (без галлюцинаций)
- context_precision     — релевантность извлечённого контекста (LLMContextPrecisionWithReference)
- context_recall        — полнота контекста относительно эталонного ответа

Судья и эмбеддинги — локальные через Ollama (qwen2.5:3b).

Запуск: uv run python scripts/evaluate_ragas.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from langchain_ollama import ChatOllama, OllamaEmbeddings
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
from corporate_assistant.config import COLLECTION_NAME, LLM_MODEL, OLLAMA_URL, TOP_K

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


def main() -> None:
    assistant = CorporateAssistant(collection_name=COLLECTION_NAME, top_k=TOP_K)

    samples: list[SingleTurnSample] = []
    for query, reference in GOLDEN:
        result = assistant.answer(query)
        contexts = [s["text"] for s in result.sources]
        # TODO refactor: instead of dropping citations, we should pass them 
        # to the RAGAS judge as a separate field
        # results.answer  may contain citations
        #  like [Файл.pdf, стр. N], which are references to sources, not facts from the context.
        # this should be changed - citations and text answer should be different fields so that 
        # the RAGAS judge can evaluate faithfulness correctly.
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

    llm = LangchainLLMWrapper(
        ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.0, request_timeout=900.0)
    )
    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="qwen2.5:3b", base_url=OLLAMA_URL)
    )

    metrics = [
        Faithfulness(llm=llm),
        LLMContextPrecisionWithReference(llm=llm),
        ContextRecall(llm=llm),
    ]

    dataset = EvaluationDataset(samples=samples)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(timeout=900, max_retries=3, max_wait=120),
    )

    print("\n=== RAGAS-результаты ===")
    print(score.to_pandas())
    print(score)

    out = ROOT / "results" / "ragas_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score._scores_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()