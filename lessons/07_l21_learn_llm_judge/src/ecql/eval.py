"""Оценка дообученной модели: синтаксис, логическая точность, галлюцинации, LLM-as-a-judge."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ecql.config import EVAL_RESULTS, save_json
from ecql.ecql_spec import describe, is_sql_style, parse_ecql
from ecql.yandex_client import YandexGPTClient, judge_ecql_pair


def generate_ecql(
    model,
    tokenizer,
    instruction: str,
    user_input: str,
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> str:
    """Один ответ модели: русский запрос -> ECQL."""
    messages = [
        {
            "role": "system",
            "content": (
                "Ты переводишь запросы с русского языка на корпоративный язык запросов ECQL. "
                "Отвечай только самим ECQL-запросом, без пояснений."
            ),
        },
        {"role": "user", "content": f"{instruction}\n\n{user_input}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    prompt = prompt.to(model.device if hasattr(model, "device") else device)
    with torch.no_grad():
        out = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][prompt.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text


def evaluate_on_testset(
    model,
    tokenizer,
    test_rows: list[dict],
    *,
    max_new_tokens: int = 128,
    device: str = "cpu",
    judge: YandexGPTClient | None = None,
    use_judge: bool = True,
    judge_max: int | None = None,
    batch_progress: bool = True,
) -> dict:
    """Полный прогон метрик на отложенной выборке.

    Метрики:
    - syntax_accuracy — доля синтаксически валидных ECQL;
    - exact_match — полное совпадение с эталоном;
    - entity/field/op/logic accuracy — совпадение смысловых компонент;
    - hallucination_rate — доля ответов с SQL-синтаксисом;
    - LLM-as-judge (YandexGPT Pro) — субъективная оценка 0..5.
    """
    preds: list[dict] = []
    n = len(test_rows)
    judge_rows = test_rows if judge_max is None else test_rows[:judge_max]

    for i, row in enumerate(test_rows):
        start = time.time()
        prediction = generate_ecql(
            model, tokenizer, row["instruction"], row["input"],
            max_new_tokens=max_new_tokens, device=device,
        )
        reference = row["output"]

        parsed = parse_ecql(prediction) if not is_sql_style(prediction) else None
        syntax_valid = parsed is not None and parsed.valid
        pred_descr = describe(prediction) if syntax_valid else {}
        ref_descr = describe(reference)

        rec = {
            "input": row["input"],
            "reference": reference,
            "prediction": prediction,
            "syntax_valid": syntax_valid,
            "sql_hallucination": is_sql_style(prediction),
            "exact_match": prediction.strip() == reference.strip(),
            "entity_match": pred_descr.get("entity") == ref_descr["entity"],
            "fields_match": pred_descr.get("fields") == ref_descr["fields"],
            "ops_match": pred_descr.get("ops") == ref_descr["ops"],
            "logic_match": pred_descr.get("logic") == ref_descr["logic"],
            "format_match": pred_descr.get("format") == ref_descr["format"],
            "inference_sec": round(time.time() - start, 2),
        }
        preds.append(rec)
        if batch_progress:
            print(f"  [{i + 1}/{n}] {rec['input'][:60]}...")

    # LLM-as-a-judge
    judge_scores: list[dict] = []
    if judge is not None and use_judge:
        print("Оценка LLM-as-a-judge (YandexGPT Pro)...")
        for row in judge_rows:
            pred = next(
                (r for r in preds if r["input"] == row["input"]), {}
            ).get("prediction", "")
            score = judge_ecql_pair(judge, row, pred)
            judge_scores.append(score)
            print(
                f"  judge {len(judge_scores)}/{len(judge_rows)}: "
                f"score={score['score']} verdict={score['verdict']}"
            )

    metrics = _aggregate(preds, judge_scores)
    result = {
        "metrics": metrics,
        "n": n,
        "judged": len(judge_scores),
        "predictions": preds,
        "judge_scores": judge_scores,
    }
    save_json(EVAL_RESULTS, result)
    return result


def _aggregate(preds: list[dict], judge_scores: list[dict]) -> dict:
    def mean(key: str) -> float:
        vals = [p[key] for p in preds]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    syntax = mean("syntax_valid")
    sql = mean("sql_hallucination")
    if judge_scores:
        judge_ok = [s for s in judge_scores if s["score"] >= 4]
        j_mean = round(sum(s["score"] for s in judge_scores) / len(judge_scores), 2)
        j_pass = round(len(judge_ok) / len(judge_scores), 4)
    else:
        j_mean, j_pass = 0.0, 0.0
    return {
        "syntax_accuracy": syntax,
        "exact_match": mean("exact_match"),
        "entity_accuracy": mean("entity_match"),
        "fields_accuracy": mean("fields_match"),
        "ops_accuracy": mean("ops_match"),
        "logic_accuracy": mean("logic_match"),
        "format_accuracy": mean("format_match"),
        "sql_hallucination_rate": sql,
        "judge_avg_score": j_mean,
        "judge_pass_rate": j_pass,
        "avg_inference_sec": round(sum(p["inference_sec"] for p in preds) / len(preds), 2),
    }


def build_eval_table(result: dict, n_rows: int = 5) -> list[dict]:
    """Собирает таблицу сравнения «вход — эталон — ответ модели» для отчёта."""
    return [
        {
            "input": p["input"],
            "reference": p["reference"],
            "prediction": p["prediction"],
            "syntax": p["syntax_valid"],
            "exact": p["exact_match"],
        }
        for p in result["predictions"][:n_rows]
    ]


def print_eval_summary(result: dict) -> None:
    m = result["metrics"]
    print("=" * 60)
    print("Сводка метрик на тестовой выборке")
    print("=" * 60)
    for k, v in m.items():
        print(f"  {k:<24}: {v}")


def load_last_eval(path: Path = EVAL_RESULTS) -> dict:
    from ecql.config import load_json

    return load_json(path)
