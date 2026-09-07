"""
vectorstore.py
A minimal persistent vector store using only NumPy - no ChromaDB, no Rust
extensions, no compiled dependencies beyond NumPy itself.

Why this exists: ChromaDB declares `tokenizers` (a Rust/PyO3 package) as a
hard install-time dependency for its default embedder - even though this
project never uses that embedder (we use Gemini or sentence-transformers
directly). On Python 3.14, `tokenizers` has no prebuilt wheel yet and PyO3
doesn't support 3.14 at all, so installing chromadb was failing outright on
Streamlit Cloud regardless of anything in our own code. Removing chromadb
removes that entire failure mode.

Storage format: one JSON file per collection containing chunk text,
metadata, and embeddings (as plain lists). Fine for the note counts this
MVP targets (tens to low thousands of chunks) - see README for the
"what I'd add" note on scaling this further.
"""

import os
import json
from typing import List, Dict, Optional

import numpy as np


class SimpleVectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "obsidian_vault"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)
        self._path = os.path.join(persist_dir, f"{collection_name}.json")
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []
        self._embeddings: Optional[np.ndarray] = None  # shape (n, dim)
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            self._ids = data.get("ids", [])
            self._texts = data.get("texts", [])
            self._metadatas = data.get("metadatas", [])
            embs = data.get("embeddings", [])
            self._embeddings = np.array(embs, dtype=np.float32) if embs else None
        except Exception:
            # corrupt/partial file - treat as empty rather than crash the app
            self._ids, self._texts, self._metadatas, self._embeddings = [], [], [], None

    def _save(self):
        data = {
            "ids": self._ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
        }
        with open(self._path, "w") as f:
            json.dump(data, f)

    def clear(self):
        self._ids, self._texts, self._metadatas, self._embeddings = [], [], [], None
        if os.path.exists(self._path):
            os.remove(self._path)

    def count(self) -> int:
        return len(self._ids)

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict], embeddings: List[List[float]]):
        self._ids.extend(ids)
        self._texts.extend(documents)
        self._metadatas.extend(metadatas)
        new_embs = np.array(embeddings, dtype=np.float32)
        if self._embeddings is None:
            self._embeddings = new_embs
        else:
            self._embeddings = np.vstack([self._embeddings, new_embs])
        self._save()

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        if self._embeddings is None or len(self._ids) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.array(query_embedding, dtype=np.float32)
        # cosine similarity -> convert to a "distance" (lower = closer) so the
        # rest of the app's scoring convention matches the old Chroma-based code
        doc_norms = np.linalg.norm(self._embeddings, axis=1)
        q_norm = np.linalg.norm(q)
        denom = doc_norms * q_norm
        denom[denom == 0] = 1e-10
        similarities = (self._embeddings @ q) / denom
        distances = 1 - similarities

        k = min(n_results, len(self._ids))
        top_idx = np.argsort(distances)[:k]

        return {
            "documents": [[self._texts[i] for i in top_idx]],
            "metadatas": [[self._metadatas[i] for i in top_idx]],
            "distances": [[float(distances[i]) for i in top_idx]],
        }
