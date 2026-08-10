import re

_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\U0001F200-\U0001F9FF]"
)

_META_PREFIX_RE = re.compile(
    r"^Вот (готовый )?(пост|текст|ответ|вариант|наш пост)[^\n]*:?\s*$",
    re.IGNORECASE,
)


def strip_code_fence(text: str) -> str:
    text = re.sub(r"^```[^\n]*\n", "", text.strip())
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def strip_meta_prefix(text: str) -> str:
    lines = text.strip().split("\n")
    while lines:
        head = lines[0].strip()
        if not head:
            lines.pop(0)
            continue
        if _META_PREFIX_RE.match(head):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


def remove_cjk(text: str) -> str:
    return _CJK_RE.sub("", text).strip()


def clean_final_text(text: str) -> str:
    cleaned = strip_code_fence(text)
    cleaned = strip_meta_prefix(cleaned)
    cleaned = remove_cjk(cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return cleaned.strip()
