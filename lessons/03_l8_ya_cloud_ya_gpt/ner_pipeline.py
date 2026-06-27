# ner_pipeline.py
import json
from pathlib import Path
from src.chunker import load_document, prepare_text, check_documents
from src.ner_client import extract_entities

DOCS_DIR  = "data/documents"
OUT_DIR   = "results"
STRATEGY  = "chunk"   # или "chunk" для длинных договоров

def merge_chunks(chunk_results: list[dict]) -> dict:
    """Объединяет результаты нескольких чанков в один объект."""
    merged = {"parties": [], "dates": {}, "amounts": None, "obligations": None}
    for r in chunk_results:
        if r.get("parties"):
            merged["parties"].extend(r["parties"])
        if r.get("dates"):
            merged["dates"].update(r["dates"])
        if r.get("amounts") and not merged["amounts"]:
            merged["amounts"] = r["amounts"]
        if r.get("obligations") and not merged["obligations"]:
            merged["obligations"] = r["obligations"]
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