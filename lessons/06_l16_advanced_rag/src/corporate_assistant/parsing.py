"""Продвинутый парсинг корпоративных документов.

- PDF: PyMuPDF (pymupdf). Извлекаем не только текст, но и метаданные страницы:
  физический номер страницы, печатный номер из колонтитула ("стр. N") и
  заголовок раздела из верхнего колонтитула.
- HTML (Wiki): BeautifulSoup. Заголовок статьи (h1) и раздела (h2) -> metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

from corporate_assistant.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_JSON,
    RAW_PDFS,
    RAW_WIKI,
)

# Верхняя/нижняя полосы страницы, где лежат колонтитулы (доля высоты страницы).
HEADER_BAND = 0.11
FOOTER_BAND = 0.92

# Печатный номер страницы вида "стр. 5".
PRINTED_PAGE_RE = re.compile(r"стр\.\s*(\d+)", re.IGNORECASE)


@dataclass
class PageContent:
    """Содержимое одной страницы PDF + извлечённые метаданные."""

    text: str
    source: str          # имя файла
    page: int | None     # печатный номер страницы (для цитирования)
    physical_page: int   # физический номер страницы в PDF
    header: str          # заголовок раздела из верхнего колонтитула
    body: str = field(default="", repr=False)

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "physical_page": self.physical_page,
            "header": self.header,
            "doc_type": "pdf",
        }


@dataclass
class Chunk:
    """Фрагмент текста с жёстко привязанным словарём метаданных."""

    text: str
    metadata: dict

    @property
    def doc_type(self) -> str:
        return self.metadata.get("doc_type", "")


def _extract_header(page: pymupdf.Page) -> str:
    """Заголовок раздела = текст в верхней полосе страницы (колонтитул)."""
    words = page.get_text("words")
    band = page.rect.height * HEADER_BAND
    top_words = sorted((w for w in words if w[3] <= band), key=lambda w: (w[1], w[0]))
    if not top_words:
        return ""
    # Схлопываем слова в одну строку по строкам (y-координатам).
    lines: list[list[str]] = []
    for word in top_words:
        if lines and abs(word[1] - lines[-1][-1][1]) < 2:
            lines[-1].append(word)
        else:
            lines.append([word])
    header = " ".join(" ".join(w[4] for w in line) for line in lines).strip()
    return header


def _extract_printed_page(page: pymupdf.Page) -> int | None:
    """Печатный номер страницы из нижнего колонтитула ("стр. N")."""
    words = page.get_text("words")
    band = page.rect.height * FOOTER_BAND
    footer_words = [w for w in words if w[1] >= band]
    footer_text = " ".join(w[4] for w in footer_words)
    match = PRINTED_PAGE_RE.search(footer_text)
    if match:
        return int(match.group(1))
    # Fallback: одиночное число в футере.
    numbers = [int(w[4]) for w in footer_words if w[4].isdigit()]
    return numbers[-1] if numbers else None


def _extract_body(page: pymupdf.Page) -> str:
    """Текст страницы без верхнего и нижнего колонтитулов."""
    words = page.get_text("words")
    header_band = page.rect.height * HEADER_BAND
    footer_band = page.rect.height * FOOTER_BAND
    body_words = [w for w in words if header_band < w[3] and w[1] < footer_band]
    if not body_words:
        return page.get_text("text")
    lines: list[list[tuple]] = []
    for word in sorted(body_words, key=lambda w: (w[1], w[0])):
        if lines and abs(word[1] - lines[-1][-1][1]) < 2:
            lines[-1].append(word)
        else:
            lines.append([word])
    return "\n".join(
        " ".join(w[4] for w in sorted(line, key=lambda w: w[0])) for line in lines
    ).strip()


def parse_pdf(path: Path) -> list[PageContent]:
    """Парсинг PDF-файла: текст + метаданные (page, header) по каждой странице.

    Страницы без верхнего колонтитула (титульный лист, оглавление) считаются
    служебными и пропускаются — они не являются источником контента.
    """
    pages: list[PageContent] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            header = _extract_header(page)
            if not header:
                continue  # титульный лист / оглавление без заголовка раздела
            printed = _extract_printed_page(page)
            body = _extract_body(page)
            if not body.strip():
                continue  # пустая страница
            # Для цитирования берём печатный номер, если он есть, иначе физический.
            page_number = printed if printed is not None else index
            pages.append(
                PageContent(
                    text=body,
                    source=path.name,
                    page=page_number,
                    physical_page=index,
                    header=header,
                )
            )
    return pages


def parse_html(path: Path) -> list[Chunk]:
    """Парсинг HTML-страницы Wiki: статья (h1) + разделы (h2) -> metadata."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    h1 = soup.find("h1")
    article_title = h1.get_text(strip=True) if h1 else path.stem.replace("_", " ")

    chunks: list[Chunk] = []
    sections = soup.find_all("section")
    if not sections:
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        if text.strip():
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        "source": article_title,
                        "page": None,
                        "header": article_title,
                        "doc_type": "wiki",
                    },
                )
            )
        return chunks

    for section in sections:
        h2 = section.find("h2")
        header = h2.get_text(strip=True) if h2 else article_title
        paragraphs = [p.get_text(" ", strip=True) for p in section.find_all("p")]
        text = "\n".join(p for p in paragraphs if p)
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    "source": article_title,
                    "page": None,
                    "header": header,
                    "doc_type": "wiki",
                },
            )
        )
    return chunks


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Разбиение текста на чанки по границам предложений с overlap."""
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) > chunk_size and buffer:
            chunks.append(buffer)
            tail = buffer[-chunk_overlap:].strip()
            buffer = f"{tail} {sentence}".strip() if tail else sentence
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    return chunks


def parse_pdf_to_chunks(path: Path, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """PDF -> чанки. Каждый чанк привязан к одной странице, поэтому page/header
    не «размазываются» по нескольким страницам."""
    chunks: list[Chunk] = []
    for page in parse_pdf(path):
        parts = split_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for part in parts:
            meta = dict(page.metadata)
            meta["page"] = page.page  # печатный номер страницы
            chunks.append(Chunk(text=part, metadata=meta))
    return chunks


def load_all_chunks(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Загрузка всех документов базы знаний -> список чанков с метаданными."""
    chunks: list[Chunk] = []

    for path in sorted(RAW_PDFS.glob("*.pdf")):
        chunks.extend(parse_pdf_to_chunks(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    for path in sorted(RAW_WIKI.glob("*.html")):
        chunks.extend(parse_html(path))

    return chunks


def save_chunks(chunks: list[Chunk]) -> None:
    """Артефакт для отчёта: дамп чанков с метаданными в JSON."""
    CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "text": chunk.text,
            "metadata": {
                key: (None if value is None else value)
                for key, value in chunk.metadata.items()
            },
        }
        for chunk in chunks
    ]
    CHUNKS_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
