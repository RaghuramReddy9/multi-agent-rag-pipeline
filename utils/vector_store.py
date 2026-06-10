"""Shared utilities for building vector stores and retrieving documents.

Uses current LangChain API:
    - langchain_huggingface for embeddings
    - langchain_chroma for vector store
    - langchain_text_splitters for chunking
    - langchain_core.documents for document types
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings
from logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=3)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embeddings instance."""
    logger.info("Initializing embeddings model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_or_load_vectorstore(
    index_path: str,
    data_path: str,
) -> Chroma:
    """Load an existing Chroma index or build a new one from the data file.

    If the index directory already exists and is non-empty, it loads from disk.
    Otherwise it reads the text file, chunks it, embeds it, and persists.
    """
    embeddings = get_embeddings()

    if os.path.isdir(index_path) and os.listdir(index_path):
        logger.info("Loading existing vector store from %s", index_path)
        return Chroma(
            persist_directory=index_path,
            embedding_function=embeddings,
        )

    logger.info("Building new vector store from %s", data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Create it or update the path in config."
        )

    # Read and chunk documents
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if not raw_text.strip():
        raise ValueError(f"Data file is empty: {data_path}")

    docs = [Document(page_content=raw_text, metadata={"source": data_path})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    logger.info(
        "Created %d chunks (size=%d, overlap=%d)",
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )

    os.makedirs(index_path, exist_ok=True)
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=index_path,
    )
    return db
