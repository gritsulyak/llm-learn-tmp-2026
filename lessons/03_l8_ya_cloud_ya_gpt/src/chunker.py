# src/chunker.py
from pathlib import Path
from typing import List

# chunk limit
# API: ~7400 токенов на запрос, ~1000 на ответ + промпт (~500).
# Остаток на документ: ~5900 токенов ≈ 4500 символов (кириллица ~1.3 сим/токен)
MAX_CHARS = 4500
# ?? maybe TODO? CHUNK_OVERLAP = 200  # перекрытие чанков, чтобы не резать сущности

def load_document(path: str) -> str:
    """Читает текстовый файл документа."""
    return Path(path).read_text(encoding="utf-8").strip()

from typing import List

def truncate_middle(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    # Если текст целиком умещается в лимит, возвращаем его одной строкой в списке
    if len(text) <= max_chars:
        return [text]
        
    # Делим лимит пополам для начала и конца документа
    chars_to_cut = max_chars # ? // 2
    
    # Выделяем сырые куски
    start_part = text[:chars_to_cut]
    end_part = text[-chars_to_cut:]
    
    # Обрезаем начало по последнему пробелу (чтобы не резать слово)
    cut_start = start_part.rfind(' ')
    clean_start = start_part[:cut_start] if cut_start != -1 else start_part
    
    # Обрезаем конец по первому пробелу (чтобы не резать слово)
    cut_end = end_part.find(' ')
    clean_end = end_part[cut_end + 1:] if cut_end != -1 else end_part
            
    return [clean_start,clean_end]


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
            parts = truncate_middle(para, max_chars)
            current = "\n[...текст обрезан...]\n".join(parts) + "\n\n"
    
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
        return truncate_middle(text)
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