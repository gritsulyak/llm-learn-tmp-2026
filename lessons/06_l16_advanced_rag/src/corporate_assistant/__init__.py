from corporate_assistant.assistant import RAGAnswer
from corporate_assistant.config import COLLECTION_NAME
from corporate_assistant.indexing import build_index, load_index, retrieve
from corporate_assistant.llm import build_llm
from corporate_assistant.parsing import (
    Chunk,
    PageContent,
    load_all_chunks,
    parse_html,
    parse_pdf,
    parse_pdf_to_chunks,
    save_chunks,
    split_text,
)
from corporate_assistant.prompts import (
    SYSTEM_PROMPT,
    format_context,
    source_label,
    user_message,
)

__all__ = [
    "Chunk",
    "PageContent",
    "RAGAnswer",
    "SYSTEM_PROMPT",
    "load_all_chunks",
    "parse_html",
    "parse_pdf",
    "parse_pdf_to_chunks",
    "save_chunks",
    "split_text",
    "build_index",
    "load_index",
    "retrieve",
    "build_llm",
    "format_context",
    "source_label",
    "user_message",
    "COLLECTION_NAME",
]

__version__ = "0.1.0"
