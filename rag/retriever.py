"""
retriever.py
Thin wrapper for top-k semantic search over the NumPy-based vector store.
"""

from typing import List, Dict
from rag.ingest import get_store_and_embedder


def retrieve(query: str, top_k: int = 5, persist_dir: str = "chroma_store", api_key: str = None) -> List[Dict]:
    store, embed_fn = get_store_and_embedder(persist_dir, api_key=api_key)
    if store is None:
        return []

    query_embedding = embed_fn.embed_query([query])[0]
    results = store.query(query_embedding, n_results=top_k)

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "score": dist,
        })
    return hits
