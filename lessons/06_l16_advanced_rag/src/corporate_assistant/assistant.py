"""RAG-ассистент: retrieval -> контекст с источниками -> LLM с системным промптом.

Ответ проходит постобработку (grounding): каждая ссылка сверяется с реальными
метаданными извлечённых чанков, выдуманные источники удаляются.
"""

from __future__ import annotations

import dataclasses

from llama_index.core.llms import ChatMessage, MessageRole

from corporate_assistant.config import TOP_K
from corporate_assistant.indexing import load_index
from corporate_assistant.llm import build_llm
from corporate_assistant.prompts import (
    SYSTEM_PROMPT,
    format_context,
    ground_citations,
    user_message,
)


@dataclasses.dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]
    context: str
    dropped_citations: list[str]


class CorporateAssistant:
    """Корпоративный ассистент с системой цитирования."""

    def __init__(self, collection_name: str, top_k: int = TOP_K) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self.index = load_index(collection_name=collection_name)
        self.llm = build_llm()

    def retrieve(self, query: str) -> list:
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        return retriever.retrieve(query)

    @staticmethod
    def _source_dicts(nodes: list) -> list[dict]:
        sources: list[dict] = []
        for node_with_score in nodes:
            node = node_with_score.node
            sources.append(
                {
                    "metadata": node.metadata,
                    "score": getattr(node_with_score, "score", None),
                    "text": node.text,
                }
            )
        return sources

    def answer(self, query: str) -> RAGAnswer:
        nodes = self.retrieve(query)
        context = format_context(nodes)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_message(context, query, nodes)),
        ]
        response = self.llm.chat(messages)
        raw_answer = str(response.message.content).strip()

        grounded, dropped = ground_citations(raw_answer, nodes)
        return RAGAnswer(
            answer=grounded,
            sources=self._source_dicts(nodes),
            context=context,
            dropped_citations=dropped,
        )


def build_assistant(collection_name: str, top_k: int = TOP_K) -> CorporateAssistant:
    return CorporateAssistant(collection_name=collection_name, top_k=top_k)
