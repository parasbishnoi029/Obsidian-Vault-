"""
ingest.py
Walks an Obsidian vault (a folder of .md files), splits each note into
overlapping chunks, embeds them, and stores everything in a persistent
Chroma collection.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict

import chromadb
from rag.embeddings import get_embedding_function

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "obsidian_vault"
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks
BACKEND_META_FILE = "backend.json"  # remembers which embedding backend built this index


def _clean_markdown(text: str) -> str:
    """Strip Obsidian-specific noise that doesn't help retrieval quality."""
    text = re.sub(r"!\[\[.*?\]\]", "", text)          # embedded images/files
    text = re.sub(r"\[\[([^\]|]+)\|?[^\]]*\]\]", r"\1", text)  # [[link|alias]] -> link
    text = re.sub(r"#(\w+)", r"\1", text)              # #tags -> tags (keep the word)
    text = re.sub(r"\n{3,}", "\n\n", text)              # collapse excess blank lines
    return text.strip()


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window chunker on characters. Good enough for markdown notes;
    swap for a token-aware splitter if notes get very long/technical."""
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_vault_files(vault_path: str) -> List[Dict]:
    """Return a list of {path, content} for every markdown file in the vault."""
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
    """
    Reads every .md file under vault_path, chunks + embeds it, and (re)builds
    the Chroma collection. Returns number of chunks indexed.
    """
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embed_fn = get_embedding_function(backend=backend, api_key=api_key)
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

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
        BATCH = 64  # keep small - Gemini embedding backend calls the API per item
        for i in range(0, len(docs), BATCH):
            collection.add(
                ids=ids[i:i + BATCH],
                documents=docs[i:i + BATCH],
                metadatas=metadatas[i:i + BATCH],
            )

    # remember which backend built this index so retriever.py can match it
    with open(os.path.join(persist_dir, BACKEND_META_FILE), "w") as f:
        json.dump({"backend": backend}, f)

    return len(docs)


def get_index_backend(persist_dir: str = CHROMA_DIR) -> str:
    meta_path = os.path.join(persist_dir, BACKEND_META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("backend", "gemini")
    return "gemini"


def get_collection(persist_dir: str = CHROMA_DIR, api_key: str = None):
    client = chromadb.PersistentClient(path=persist_dir)
    backend = get_index_backend(persist_dir)
    try:
        embed_fn = get_embedding_function(backend=backend, api_key=api_key)
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        return None
