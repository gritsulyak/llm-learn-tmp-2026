"""
Пайплайн загрузки документов: PDF/TXT -> чанки -> эмбеддинги -> Qdrant.

Запуск:
    python -m src.ingest
или (после активации venv):
    python src/ingest.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

from src.config import (
    DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    QDRANT_URL, QDRANT_COLLECTION, QDRANT_TEST_COLLECTION, TEST_DOCS_DIR,
)

def load_documents(docs_dir: Path):
    """Загружает все PDF и TXT документы из указанной папки (рекурсивно)."""
    docs = []

    print(f"Гружу из: {docs_dir}")
    pdf_loader = DirectoryLoader(
    str(docs_dir), glob="**/*.pdf", loader_cls=PyPDFium2Loader,
    show_progress=True,
    silent_errors=True,
    )
    docs.extend(pdf_loader.load())

    txt_docs = []
    for p in docs_dir.rglob("*.txt"):
        doc = load_txt_with_fallback_encoding(p)
        if doc is not None:
            txt_docs.append(doc)
    docs.extend(txt_docs)

    for d in docs:
        d.metadata["source_file"] = Path(d.metadata.get("source", "unknown")).name

    print(f"Загружено документов: {len(docs)}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Получено чанков: {len(chunks)} (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_embeddings():
    """Мультиязычная CPU-модель эмбеддингов."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_collection(client: QdrantClient, embeddings, collection_name: str, sample_text: str = "probe"):
    if client.collection_exists(collection_name):
        return
    vector_size = len(embeddings.embed_query(sample_text))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Создана коллекция '{collection_name}' (dim={vector_size})")

def load_txt_with_fallback_encoding(path: Path):
    encodings = ["utf-8", "cp1251", "koi8-r", "cp866", "latin-1"]
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            return Document(page_content=text, metadata={"source": str(path)})
        except (UnicodeDecodeError, LookupError):
            continue
    print(f"⚠️ Не удалось прочитать файл (пропущен): {path}")
    return None




def run_ingest(docs_dir: Path, collection_name: str):
    if not docs_dir.exists():
        raise FileNotFoundError(f"Папка с документами не найдена: {docs_dir}")

    docs = load_documents(docs_dir)
    if not docs:
        print(f"Документы не найдены в {docs_dir}.")
        return

    chunks = split_documents(docs)
    embeddings = get_embeddings()

    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, embeddings, collection_name)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks, batch_size=64)
    print(f"✅ {len(chunks)} чанков сохранено в Qdrant (коллекция '{collection_name}').")


def main():
    import sys
    if "--test" in sys.argv:
        run_ingest(TEST_DOCS_DIR, QDRANT_TEST_COLLECTION)
    else:
        run_ingest(DOCS_DIR, QDRANT_COLLECTION)

if __name__ == "__main__":
    main()
