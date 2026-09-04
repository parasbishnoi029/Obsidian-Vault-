"""
embeddings.py
Pluggable embedding backends.

Why this file exists: the first draft of this project embedded locally
with sentence-transformers (pulls in PyTorch). That's fine on a dev
machine but risks failing / OOM-ing on Streamlit Community Cloud's free
tier (1GB RAM) - which would break the "live demo must be working"
requirement. So the default backend is Gemini's hosted embedding API
(no heavy local dependency, no model download). Local embeddings stay
available as an opt-in for offline dev use.

MODEL/SDK NOTES (verified live against Google's docs, not assumed from
training data - these change fast):
- text-embedding-004 was fully shut down Jan 14, 2026. Do not use it.
- gemini-embedding-001 is the current generally-available embedding model.
- The old `google-generativeai` package is end-of-life. This uses the
  current `google-genai` package (`from google import genai`).
"""

import os
import time
from chromadb import EmbeddingFunction, Documents, Embeddings

GEMINI_EMBED_MODEL = "gemini-embedding-001"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Uses Gemini's hosted embedding API. Distinguishes document vs query
    embeddings via task_type - measurably improves retrieval quality for
    asymmetric search (short query -> longer passage)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini embedding backend.")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)
        self._types = __import__("google.genai.types", fromlist=["types"])

    def _embed(self, texts, task_type: str):
        out = []
        for t in texts:
            for attempt in range(3):
                try:
                    resp = self._client.models.embed_content(
                        model=GEMINI_EMBED_MODEL,
                        contents=t,
                        config=self._types.EmbedContentConfig(task_type=task_type),
                    )
                    out.append(resp.embeddings[0].values)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        return out

    def __call__(self, input: Documents) -> Embeddings:
        return self._embed(input, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, input: Documents) -> Embeddings:
        return self._embed(input, task_type="RETRIEVAL_QUERY")

    def name(self) -> str:
        return "gemini-embedding-001"


def get_local_embedding_function():
    """Optional offline backend - only imports sentence-transformers when
    actually selected, so the Gemini path never pays that dependency cost."""
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=LOCAL_EMBED_MODEL)


def get_embedding_function(backend: str = "gemini", api_key: str = None):
    """backend: 'gemini' (default, recommended for deployment) or 'local'."""
    if backend == "local":
        return get_local_embedding_function()
    return GeminiEmbeddingFunction(api_key=api_key)
