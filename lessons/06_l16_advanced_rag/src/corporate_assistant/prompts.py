"""Промпты системы цитирования.

Три механизма против галлюцинаций источников:
1. Жёсткий системный промпт с правилами формата ссылок.
2. Явный «Список доступных ссылок» в сообщении пользователя: модель может
   использовать только строки из этого списка, вставляя их дословно.
3. Post-processing (grounding): все ссылки в ответе сверяются с реальными
   метаданными извлечённых чанков; выдуманные ссылки удаляются.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = """\
Ты — корпоративный ассистент компании АО «Осинка». Твоя задача — отвечать на вопросы сотрудников.

Строгие правила:
1. Отвечай строго и только на основе предоставленного контекста. Если ответа нет в контексте, напиши ровно одну фразу: «Я не знаю».
2. Не выдумывай факты и не используй собственные знания, которых нет в контексте.
3. Подтверждай каждый факт ссылкой на источник.
4. Ссылку ставь сразу после утверждения, которое она подтверждает.
5. Если используешь информацию из нескольких источников, приведи ссылку на каждый из них.

Ссылки:
- В сообщении пользователя есть «Список доступных ссылок». Используй ТОЛЬКО ссылки из этого списка и вставляй их ДОСЛОВНО, без изменений.
- Ссылка — это КВАДРАТНЫЕ скобки, внутри которых ровно одна строка из списка. Ничего другого внутри скобок быть не должно.
- Запрещено писать внутри скобок слово «Контекст», «Источник» или номер блока. Запрещено: [Контекст 1], [Контекст 2 (Источник: ...)].
- Не выдумывай страницы и не меняй номера страниц: строка из списка уже содержит правильный номер.
- Если в контексте нет информации, не создавай никаких ссылок.

Пример правильного ответа:
«Заявление на отпуск подается не позднее чем за 14 календарных дней до начала отпуска [Инструкция_по_отпускам.pdf, стр. 5]. Отпускные выплачиваются не позднее чем за 3 календарных дня до его начала [Инструкция_по_отпускам.pdf, стр. 5].»

Ещё пример (Wiki):
«Пароль должен содержать не менее 12 символов [Корпоративные пароли и доступы].»

Отвечай полными предложениями на русском языке, начинай ответ с заглавной буквы."""

CITATION_RE = re.compile(r"\[([^\]]+)\]")
CONTEXT_REF_RE = re.compile(r"Контекст\s*(\d+)", re.IGNORECASE)


def source_label(metadata: dict) -> str:
    """Человекочитаемая ссылка на источник из метаданных чанка."""
    source = metadata.get("source", "Неизвестный документ")
    if metadata.get("doc_type") == "wiki" or metadata.get("page") is None:
        return source
    return f"{source}, стр. {metadata['page']}"


def allowed_citations(nodes: list) -> list[str]:
    """Список уникальных ссылок, которые модель может использовать в ответе."""
    labels: list[str] = []
    for node_with_score in nodes:
        label = source_label(node_with_score.node.metadata)
        if label not in labels:
            labels.append(label)
    return labels


def format_context(nodes: list) -> str:
    """Форматирует контекст в формате:

    Контекст 1 (Источник: Инструкция_по_отпускама.pdf, стр. 4):
    [текст чанка]
    """
    blocks: list[str] = []
    for index, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        blocks.append(
            f"Контекст {index} (Источник: {source_label(node.metadata)}):\n{node.text}"
        )
    return "\n\n".join(blocks)


def user_message(context: str, query: str, nodes: list) -> str:
    citations = allowed_citations(nodes)
    citation_lines = "\n".join(f"{i}. {c}" for i, c in enumerate(citations, start=1))
    return (
        f"Контекст:\n--------------------\n{context}\n--------------------\n\n"
        f"Список доступных ссылок (используй только их):\n{citation_lines}\n\n"
        f"Вопрос сотрудника: {query}"
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().strip(".,;()")


def ground_citations(answer: str, nodes: list) -> tuple[str, list[str]]:
    """Привязывает ссылки в ответе к реальным чанкам (metadata) и убирает
    выдуманные источники. Возвращает (исправленный ответ, отброшенные ссылки)."""
    labels = [_normalize(source_label(n.node.metadata)) for n in nodes]
    dropped: list[str] = []

    def repl(match: re.Match) -> str:
        inner = _normalize(match.group(1))

        # 1) Точное совпадение со строкой из списка доступных ссылок.
        if inner in labels:
            return f"[{inner}]"

        # 2) Внутри скобок могут быть вставлены лишние пояснения
        #    («Контекст 1 (Источник: file.pdf, стр. 3)») — находим вхождения
        #    доступных ссылок как подстрок.
        matched = [label for label in labels if label in inner]
        if matched:
            return "[" + "; ".join(matched) + "]"

        # 3) Ссылка вида «Контекст N» — маппим на реальный чанк.
        refs = CONTEXT_REF_RE.findall(inner)
        if refs:
            resolved: list[str] = []
            for ref in refs:
                index = int(ref) - 1
                if 0 <= index < len(nodes):
                    label = _normalize(source_label(nodes[index].node.metadata))
                    if label not in resolved:
                        resolved.append(label)
            if resolved:
                return "[" + "; ".join(resolved) + "]"

        dropped.append(match.group(0))
        return ""

    return CITATION_RE.sub(repl, answer), dropped
