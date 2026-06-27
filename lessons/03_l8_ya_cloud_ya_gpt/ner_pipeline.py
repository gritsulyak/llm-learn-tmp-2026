# ner_pipeline.py
import json
from pathlib import Path
from src.chunker import load_document, prepare_text, check_documents
from src.ner_client import extract_entities

DOCS_DIR  = "data/documents"
OUT_DIR   = "results"
STRATEGY  = "chunk"   # или "chunk" для длинных договоров

def merge_chunks(chunk_results: list[dict]) -> dict:
    """Объединяет результаты нескольких чанков без потери данных."""
    merged = {"parties": [], "dates": {}, "amounts": [], "obligations": []}

    for r in chunk_results:
        # parties — всегда список dict
        if r.get("parties"):
            parties = r["parties"]
            if isinstance(parties, list):
                merged["parties"].extend(parties)

        # dates — объединяем, НЕ перезаписывая уже найденные ключи
        if r.get("dates") and isinstance(r["dates"], dict):
            for k, v in r["dates"].items():
                if v is not None and k not in merged["dates"]:
                    merged["dates"][k] = v

        # amounts — копим все найденные суммы (строку или список)
        if r.get("amounts") is not None:
            val = r["amounts"]
            if isinstance(val, list):
                merged["amounts"].extend([v for v in val if v])
            elif isinstance(val, str) and val.strip():
                merged["amounts"].append(val)

        # obligations — то же самое: строка или список
        if r.get("obligations") is not None:
            val = r["obligations"]
            if isinstance(val, list):
                merged["obligations"].extend([v for v in val if v])
            elif isinstance(val, str) and val.strip():
                merged["obligations"].append(val)

    # Дедупликация сторон по ИНН (или name если ИНН нет)
    seen, unique = set(), []
    for p in merged["parties"]:
        if not isinstance(p, dict):
            continue
        key = p.get("inn") or p.get("name")
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    merged["parties"] = unique

    # Дедупликация amounts и obligations с сохранением порядка
    merged["amounts"]     = list(dict.fromkeys(merged["amounts"]))     or None
    merged["obligations"] = list(dict.fromkeys(merged["obligations"])) or None

    # amounts: если одна строка — вернуть строку, иначе список (для обратной совместимости)
    if isinstance(merged["amounts"], list) and len(merged["amounts"]) == 1:
        merged["amounts"] = merged["amounts"][0]

    return merged

def merge_chunksv0(chunk_results: list[dict]) -> dict:
    """Объединяет результаты нескольких чанков в один объект."""
    merged = {"parties": [], "dates": {}, "amounts": None, "obligations": []}
    for r in chunk_results:
        if r.get("parties"):
            merged["parties"].extend(r["parties"])
        if r.get("dates"):
            merged["dates"].update(r["dates"])
        if r.get("amounts") and not merged["amounts"]:
            merged["amounts"] = r["amounts"]
        if r.get("obligations"):
            merged["obligations"].extend(r["obligations"])
    # Дедупликация сторон по ИНН
    seen, unique = set(), []
    for p in merged["parties"]:
        key = p.get("inn", p.get("name"))
        if key not in seen:
            seen.add(key); unique.append(p)
    merged["parties"] = unique
    return merged


def process_document(path: str) -> dict:
    print(f"\n [=] Обрабатываю: {Path(path).name}")
    text = load_document(path)
    chunks = prepare_text(text, strategy=STRATEGY)
    print(f"   Длина: {len(text):,} симв. → {len(chunks)} чанк(ов)")

    results = []
    for i, chunk in enumerate(chunks):
        print(f"   Чанк {i+1}/{len(chunks)}...")
        try:
            result = extract_entities(chunk)
            results.append(result)
        except Exception as e:
            print(f" !!!  Ошибка в чанке {i+1}: {e}")
            results.append({"error": str(e)})

    return merge_chunks(results) if len(results) > 1 else results

def main():
    Path(OUT_DIR).mkdir(exist_ok=True)

    # 1. Диагностика документов
    print("=== Анализ документов ===")
    check_documents(DOCS_DIR)

    # 2. Обработка каждого документа
    print("\n=== NER Пайплайн ===")
    docs = sorted(Path(DOCS_DIR).glob("*.txt"))
    if not docs:
        print(f"Добавь .txt файлы в {DOCS_DIR}/")
        return

    summary = {}
    for doc_path in docs:
        result = process_document(str(doc_path))
        out_file = Path(OUT_DIR) / f"{doc_path.stem}_ner.json"
        out_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        summary[doc_path.name] = result
        print(f"[V] Сохранено -> {out_file}")

    # 3. Сводный файл
    summary_path = Path(OUT_DIR) / "_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n [w] Сводка: {summary_path}")

if __name__ == "__main__":
    main()