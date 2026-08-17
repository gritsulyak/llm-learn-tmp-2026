"""Тесты парсинга PDF/HTML и системы цитирования (без эмбеддингов и Qdrant)."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import NodeWithScore, TextNode

from corporate_assistant.parsing import (
    load_all_chunks,
    parse_html,
    parse_pdf,
    split_text,
)
from corporate_assistant.prompts import (
    ground_citations,
    source_label,
)

ROOT = Path(__file__).resolve().parents[1]
PDFS = ROOT / "data" / "raw" / "pdfs"
WIKI = ROOT / "data" / "raw" / "wiki"


def _nodes(metas: list[dict]) -> list[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text=f"text{i}", metadata=meta))
        for i, meta in enumerate(metas)
    ]


class TestPdfParsing:
    def test_vacation_metadata(self):
        pages = parse_pdf(PDFS / "Инструкция_по_отпускам.pdf")
        by_page = {p.physical_page: p for p in pages}
        # Титул и оглавление (без заголовка раздела) пропущены.
        assert 1 not in by_page
        assert 2 not in by_page
        # Раздел «Заявление на отпуск» на физической и печатной странице 5.
        page5 = by_page[5]
        assert page5.header == "Заявление на отпуск"
        assert page5.page == 5
        assert page5.physical_page == 5
        assert "14 календарных дней" in page5.text

    def test_printed_page_offset(self):
        """Положение_о_командировках.pdf: печатная нумерация сдвинута титулом
        и оглавлением на 2 (физическая стр. 5 печатает «стр. 3»)."""
        pages = parse_pdf(PDFS / "Положение_о_командировках.pdf")
        by_page = {p.physical_page: p for p in pages}
        assert 1 not in by_page
        assert by_page[5].header == "Расходы и суточные"
        assert by_page[5].page == 3  # печатный номер, а не физический 5

    def test_all_pdf_pages_have_metadata(self):
        for path in PDFS.glob("*.pdf"):
            for page in parse_pdf(path):
                assert page.source == path.name
                assert page.page is not None
                assert page.header


class TestWikiParsing:
    def test_article_and_sections(self):
        chunks = parse_html(WIKI / "Корпоративные_пароли.html")
        assert chunks
        headers = {c.metadata["header"] for c in chunks}
        assert "Требования к паролю" in headers
        assert "Смена пароля и двухфакторная аутентификация" in headers
        for chunk in chunks:
            assert chunk.metadata["doc_type"] == "wiki"
            assert chunk.metadata["source"] == "Корпоративные пароли и доступы"


class TestChunking:
    def test_chunks_do_not_cross_pages(self):
        chunks = load_all_chunks(chunk_size=300, chunk_overlap=30)
        pdf_chunks = [c for c in chunks if c.doc_type == "pdf"]
        assert pdf_chunks
        for chunk in pdf_chunks:
            assert chunk.metadata.get("page") is not None
            assert chunk.metadata.get("header")

    def test_split_text_respects_boundaries(self):
        parts = split_text("Первый предложение. Второе предложение? Третье!", chunk_size=30, chunk_overlap=5)
        assert parts
        assert all(" ".join(parts).find(p) >= 0 for p in parts)


class TestCitations:
    def test_source_label(self):
        assert source_label({"source": "a.pdf", "page": 4, "doc_type": "pdf"}) == "a.pdf, стр. 4"
        assert source_label({"source": "Wiki статья", "doc_type": "wiki"}) == "Wiki статья"

    def test_grounding_keeps_exact(self):
        nodes = _nodes([{"source": "a.pdf", "page": 4, "doc_type": "pdf"}])
        out, dropped = ground_citations("Факт [a.pdf, стр. 4].", nodes)
        assert out == "Факт [a.pdf, стр. 4]."
        assert dropped == []

    def test_grounding_resolves_context_number(self):
        nodes = _nodes(
            [
                {"source": "a.pdf", "page": 4, "doc_type": "pdf"},
                {"source": "Wiki", "doc_type": "wiki"},
            ]
        )
        out, dropped = ground_citations("Факт [Контекст 2].", nodes)
        assert out == "Факт [Wiki]."
        assert dropped == []

    def test_grounding_drops_hallucinated(self):
        nodes = _nodes([{"source": "a.pdf", "page": 4, "doc_type": "pdf"}])
        out, dropped = ground_citations("Зарплата [несуществующий.pdf, стр. 99].", nodes)
        assert "несуществующий.pdf" not in out
        assert dropped == ["[несуществующий.pdf, стр. 99]"]
