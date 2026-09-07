"""
embeddings.py
Pluggable embedding backends.

Default is Gemini's hosted embedding API - lightweight, no heavy local
dependency, no model download, so it doesn't risk failing on constrained or
version-mismatched cloud environments. Local sentence-transformers is kept
as an opt-in for offline dev use.

MODEL/SDK NOTES (verified live, not assumed from training data):
- text-embedding-004 was fully shut down Jan 14, 2026 - do not use it.
- gemini-embedding-001 is the current generally-available embedding model.
- The old `google-generativeai` package is end-of-life - this uses the
  current `google-genai` package instead.

NOTE: this file intentionally has no dependency on chromadb. Earlier
versions subclassed chromadb.EmbeddingFunction, but chromadb itself pulls in
`tokenizers` (a Rust/PyO3 package) as a hard dependency even when unused,
which broke installs on newer Python versions before PyO3 caught up. This
project now uses its own minimal interface instead: any object with a
__call__(list[str]) -> list[list[float]] method (for documents) and an
embed_query(list[str]) -> list[list[float]] method (for queries).
"""

import os
import time

GEMINI_EMBED_MODEL = "gemini-embedding-001"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"


class GeminiEmbeddingFunction:
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

    def __call__(self, input):
        return self._embed(input, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, input):
        return self._embed(input, task_type="RETRIEVAL_QUERY")


class LocalEmbeddingFunction:
    """Offline backend using sentence-transformers directly (no chromadb
    dependency). Only imported when actually selected."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(LOCAL_EMBED_MODEL)

    def __call__(self, input):
        return self._model.encode(input, convert_to_numpy=True).tolist()

    def embed_query(self, input):
        # symmetric model - same encoding path for queries and documents
        return self.__call__(input)


def get_embedding_function(backend: str = "gemini", api_key: str = None):
    """backend: 'gemini' (default, recommended for deployment) or 'local'."""
    if backend == "local":
        return LocalEmbeddingFunction()
    return GeminiEmbeddingFunction(api_key=api_key)
