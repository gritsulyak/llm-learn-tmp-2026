"""Векторизация чанков и сборка векторной базы (Qdrant) через LlamaIndex."""

from __future__ import annotations

import time

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from corporate_assistant.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_URL,
    QDRANT_URL,
    TOP_K,
)
from corporate_assistant.parsing import Chunk


def select_device() -> str:
    """Ускоряем embeddings, если доступен GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def make_embed_model():
    device = select_device()
    print(f"Embedding model: {EMBEDDING_MODEL} on {device}")
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device=device, normalize=True)


def configure_settings() -> None:
    Settings.embed_model = make_embed_model()
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=LLM_TEMPERATURE,
        request_timeout=600.0,
    )
    # Чанки уже нарезаны в parsing.py: разбивать повторно не нужно.
    Settings.node_parser = SimpleNodeParser.from_defaults()


def chunks_to_nodes(chunks: list[Chunk]) -> list[TextNode]:
    return [
        TextNode(text=chunk.text, metadata=dict(chunk.metadata)) for chunk in chunks
    ]


def build_index(chunks: list[Chunk], *, reset: bool = False, collection_name: str = COLLECTION_NAME) -> VectorStoreIndex:
    """Собирает Qdrant-коллекцию: векторы + payload с метаданными каждого чанка."""
    client = qdrant_client()

    if client.collection_exists(collection_name):
        if not reset:
            print(f"Коллекция {collection_name!r} уже существует, пересборка пропущена. Используйте --reset.")
        else:
            print(f"Удаляю коллекцию {collection_name!r}...")
            client.delete_collection(collection_name)

    configure_settings()
    nodes = chunks_to_nodes(chunks)
    print(f"Чанков к индексации: {len(nodes)}")

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    started = time.monotonic()
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    elapsed = time.monotonic() - started
    print(f"Индекс собран за {elapsed:.1f} c: Qdrant collection={collection_name!r}")
    return index


def load_index(collection_name: str = COLLECTION_NAME) -> VectorStoreIndex:
    """Подключение к существующей коллекции без повторной индексации."""
    client = qdrant_client()
    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Коллекция {collection_name!r} не найдена. Сначала запустите build_index."
        )
    configure_settings()
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def retrieve(query: str, top_k: int = TOP_K, collection_name: str = COLLECTION_NAME) -> list:
    """Retriever: топ-K релевантных чанков вместе с метаданными."""
    index = load_index(collection_name=collection_name)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)
