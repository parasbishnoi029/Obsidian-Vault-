"""
ingest.py
Walks an Obsidian vault (a folder of .md files), splits each note into
overlapping chunks, embeds them, and stores everything in a persistent
NumPy-based vector store (see vectorstore.py for why this isn't ChromaDB).
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict

from rag.embeddings import get_embedding_function
from rag.vectorstore import SimpleVectorStore

CHROMA_DIR = "chroma_store"  # kept the name for continuity with earlier versions
COLLECTION_NAME = "obsidian_vault"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BACKEND_META_FILE = "backend.json"


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[([^\]|]+)\|?[^\]]*\]\]", r"\1", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_vault_files(vault_path: str) -> List[Dict]:
    files = []
    for p in Path(vault_path).rglob("*.md"):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"skipped {p}: {e}")
            continue
        files.append({"path": str(p.relative_to(vault_path)), "content": content})
    return files


def build_index(vault_path: str, persist_dir: str = CHROMA_DIR,
                 backend: str = "gemini", api_key: str = None) -> int:
    os.makedirs(persist_dir, exist_ok=True)

    embed_fn = get_embedding_function(backend=backend, api_key=api_key)
    store = SimpleVectorStore(persist_dir, COLLECTION_NAME)
    store.clear()  # full rebuild each time, same behaviour as before

    files = load_vault_files(vault_path)
    if not files:
        return 0

    ids, docs, metadatas = [], [], []
    for f in files:
        cleaned = _clean_markdown(f["content"])
        for i, chunk in enumerate(_chunk_text(cleaned)):
            chunk_id = hashlib.md5(f"{f['path']}-{i}".encode()).hexdigest()
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append({"source": f["path"], "chunk_index": i})

    if docs:
        BATCH = 64
        for i in range(0, len(docs), BATCH):
            batch_docs = docs[i:i + BATCH]
            batch_embeddings = embed_fn(batch_docs)  # __call__ = document embeddings
            store.add(
                ids=ids[i:i + BATCH],
                documents=batch_docs,
                metadatas=metadatas[i:i + BATCH],
                embeddings=batch_embeddings,
            )

    with open(os.path.join(persist_dir, BACKEND_META_FILE), "w") as f:
        json.dump({"backend": backend}, f)

    return len(docs)


def get_index_backend(persist_dir: str = CHROMA_DIR) -> str:
    meta_path = os.path.join(persist_dir, BACKEND_META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("backend", "gemini")
    return "gemini"


def get_store_and_embedder(persist_dir: str = CHROMA_DIR, api_key: str = None):
    """Returns (store, embed_fn) or (None, None) if nothing indexed yet."""
    store = SimpleVectorStore(persist_dir, COLLECTION_NAME)
    if store.count() == 0:
        return None, None
    backend = get_index_backend(persist_dir)
    embed_fn = get_embedding_function(backend=backend, api_key=api_key)
    return store, embed_fn
