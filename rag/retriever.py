"""
retriever.py
Thin wrapper around the Chroma collection for top-k semantic search.
"""

from typing import List, Dict
from rag.ingest import get_collection


def retrieve(query: str, top_k: int = 5, persist_dir: str = "chroma_store", api_key: str = None) -> List[Dict]:
    """
    Returns a list of {text, source, score} for the top_k most relevant chunks.
    score is a distance (lower = more similar) as returned by Chroma.
    """
    collection = get_collection(persist_dir, api_key=api_key)
    if collection is None or collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=top_k)

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
