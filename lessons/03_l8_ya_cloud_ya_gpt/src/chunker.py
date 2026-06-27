# src/chunker.py
from pathlib import Path
from typing import List

# chunk limit
# API: ~7400 токенов на запрос, ~1000 на ответ + промпт (~500).
# Остаток на документ: ~5900 токенов ≈ 4500 символов (кириллица ~1.3 сим/токен)
MAX_CHARS = 4500
CHUNK_OVERLAP = 200  # перекрытие чанков, чтобы не резать сущности

def load_document(path: str) -> str:
    """Читает текстовый файл документа."""
    return Path(path).read_text(encoding="utf-8").strip()

def truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """Простая обрезка: берём первые max_chars символов.
    Подходит если реквизиты/даты обычно в начале документа."""
    if len(text) <= max_chars:
        return text
    # Обрезаем по последнему пробелу, чтобы не резать слово
    cut = text[:max_chars].rfind(" ")
    return text[:cut] + "\n[...текст обрезан...]"

def chunk_by_paragraphs(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """Разбивает текст на смысловые части по абзацам.
    Используй когда важно обработать документ целиком (длинный договор)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            # Если один абзац больше лимита — трункируем его
            current = truncate(para, max_chars) + "\n\n"
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks

def prepare_text(text: str, strategy: str = "truncate") -> List[str]:
    """
    strategy='truncate' — быстро, подходит для большинства договоров
                          (реквизиты обычно в начале)
    strategy='chunk'    — полная обработка, нужно объединять результаты чанков
    """
    if strategy == "truncate":
        return [truncate(text)]
    elif strategy == "chunk":
        return chunk_by_paragraphs(text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def check_documents(folder: str = "data/documents") -> None:
    """Диагностика: выводит длину каждого документа и рекомендует стратегию."""
    docs = sorted(Path(folder).glob("*.txt"))
    if not docs:
        print(f"Нет .txt файлов в {folder}/")
        return
    
    print(f"{'Файл':<30} {'Символов':>10} {'Токенов ~':>12} {'Стратегия':>12}")
    print("-" * 70)
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        chars = len(text)
        tokens_est = chars // 1.3  # грубая оценка для кириллицы
        strategy = "ok" if chars <= MAX_CHARS else ("truncate" if chars <= MAX_CHARS * 3 else "chunk")
        print(f"{doc.name:<30} {chars:>10,} {tokens_est:>12,.0f} {strategy:>12}")

if __name__ == "__main__":
    check_documents()