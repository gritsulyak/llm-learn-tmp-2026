"""
RAG-цепочка: связывает Qdrant (retriever) и локальную LLM (Ollama) через LCEL.

Основано на паттернах из учебного ноутбука Arkhitektura_dialogovykh_sistem_v_LangChain
(BaseChatBot / create_llm / LCEL runnable chain).
"""
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("rag_llm")

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import (
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, EMBEDDING_MODEL,
    QDRANT_URL, QDRANT_COLLECTION, TOP_K, SYSTEM_PROMPT_TEMPLATE,
)


def create_llm(model_name: str = OLLAMA_LLM_MODEL, temperature: float = 0.2):
    """Создает клиент локальной LLM через Ollama."""
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=1024,   # лимит длины ответа
        num_ctx=2048,      # меньше контекст -> быстрее на CPU
        reasoning=False,   # отключает "Thinking..." у qwen3.5
    )

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_retriever(top_k: int = TOP_K, collection_name: str = QDRANT_COLLECTION):
    client = QdrantClient(url=QDRANT_URL)
    embeddings = get_embeddings()
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

def format_docs(docs) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
        parts.append(f"[Источник: {src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


class RagBot:
    """RAG-бот: аналог BaseChatBot из ноутбука, но с retrieval-компонентом."""
    def __init__(self, model_name: str = OLLAMA_LLM_MODEL, top_k: int = TOP_K, temperature: float = 0.2, collection_name: str = QDRANT_COLLECTION):
        self.model_name = model_name
        self.top_k = top_k
        self.collection_name = collection_name
        self.llm = create_llm(model_name, temperature)
        self.retriever = get_retriever(top_k, collection_name=self.collection_name)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEMPLATE),
            ("human", "{question}"),
        ])

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> Dict:
        """Возвращает ответ + найденные источники + время генерации (latency)."""
        t0 = time.time()
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        full_prompt = self.prompt.format(context=context, question=question)

        logger.info(f"LLM INPUT >>> {full_prompt}")

        answer = self.chain.invoke(question)
        latency = time.time() - t0

        logger.info(f"LLM OUTPUT <<< {answer}")
        logger.info(f"LLM LATENCY = {round(latency, 2)} sec")

        # Пытаемся достать реально использованные источники из ответа модели
        used_sources = []
        clean_answer = answer
        if "ИСПОЛЬЗОВАНО:" in answer:
            clean_answer, used_line = answer.split("ИСПОЛЬЗОВАНО:", 1)
            clean_answer = clean_answer.strip()
            used_sources = [s.strip() for s in used_line.strip().split(",") if s.strip()]

        # fallback: если модель не проставила метку - показываем все найденные (как раньше)
        all_sources = sorted({
            f"{d.metadata.get('source_file', 'unknown')} (стр. {d.metadata.get('page_number', '?')})"
            for d in docs
        })

        return {
            "answer": clean_answer,
            "sources": used_sources if used_sources else all_sources,
            "latency_sec": round(latency, 2),
            "num_chunks": len(docs),
        }

    def set_top_k(self, top_k: int):
        """Позволяет менять Top-K на лету для экспериментов (см. TASK_RU.md п.4)."""
        self.top_k = top_k
        self.retriever = get_retriever(top_k, collection_name=self.collection_name)
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )


if __name__ == "__main__":
    bot = RagBot()
    while True:
        q = input("\nВопрос (или 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        result = bot.ask(q)
        print(f"\nОтвет: {result['answer']}")
        print(f"Источники: {result['sources']}")
        print(f"Время: {result['latency_sec']} сек | чанков: {result['num_chunks']}")
