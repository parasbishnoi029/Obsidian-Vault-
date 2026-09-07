# Obsidian Vault RAG Knowledge Assistant

A retrieval-augmented generation (RAG) assistant that answers questions
grounded in your own Obsidian vault (a folder of markdown notes) — built as
a 1-week MVP for a Generative AI Developer Intern take-home.

## What it does

1. Point it at a folder of `.md` notes (a sample vault ships with the repo,
   or upload your own).
2. It chunks each note, embeds the chunks, and stores them in a lightweight
   NumPy-based vector store.
3. Ask a question in the chat box. It retrieves the most relevant chunks
   and asks Gemini to answer **using only that context**, citing which
   note(s) the answer came from.
4. If no Gemini key is set, it still runs for retrieval — see limitations
   below on when generation needs a key.

## Why this approach

I picked RAG over the scraping/agent options mainly for reliability under a
hard deadline. Mid-build I also had to make a second reliability call: I
originally used ChromaDB as the vector store, but it declares `tokenizers`
(a Rust/PyO3 package) as a hard dependency even though this project never
uses the embedder that needs it. On newer Python versions where PyO3 hasn't
caught up yet, that dependency fails to build and breaks the whole install
— independent of anything in this project's own code. Rather than fight a
transitive dependency I don't need, I replaced ChromaDB with a small
NumPy-only vector store (`rag/vectorstore.py`). Same retrieval behavior,
zero compiled/Rust dependencies, works on any Python version NumPy supports.

That decision — swapping out a dependency instead of patching around its
failure — is itself part of what I'd want evaluated: recognizing when a
library choice becomes the wrong choice under new constraints matters more
than knowing the original API.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| UI | Streamlit | fastest way to ship a usable chat UI + free one-click deploy |
| Embeddings | Gemini `gemini-embedding-001` (default), local `sentence-transformers` (optional) | Gemini backend has no heavy dependency, so it doesn't risk OOM-ing on free-tier cloud hosting |
| Vector store | Custom NumPy-based store (`rag/vectorstore.py`), JSON-persisted | Zero compiled dependencies — deploys cleanly on any Python version, including brand-new ones where compiled-wheel ecosystems haven't caught up yet |
| Generation | `gemini-flash-latest` | fast, generous free tier, Google's stable alias so it doesn't break on the next model deprecation |
| Chunking | custom sliding-window splitter (800 chars, 150 overlap) | simple, no extra dependency, easy to explain/tune |

### A note on why the model names look specific

Gemini model names churn fast — while building this I found that
`gemini-1.5-flash` (what most tutorials still reference) is no longer
usable in new projects, and `text-embedding-004` was fully shut down on
Jan 14, 2026. The old `google-generativeai` Python package is also
end-of-life, replaced by `google-genai`. This project uses the current
package and Google's stable `-latest` alias rather than a pinned dated
model name, specifically so it doesn't quietly break between when I
submit this and when it gets evaluated.

## Architecture

```
data/sample_vault/*.md  ──┐
                           │
                    rag/ingest.py       (chunk + call embeddings.py + store)
                           │
                  rag/embeddings.py     (Gemini API or local sentence-transformers)
                           │
                  rag/vectorstore.py    (NumPy cosine-similarity store, JSON-persisted)
                           │
                  rag/retriever.py      (top-k semantic search)
                           │
                  rag/generator.py      (build grounded prompt → Gemini)
                           │
                        app.py          (Streamlit chat UI)
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the sidebar:
1. Paste a Gemini API key (free at https://aistudio.google.com/apikey)
2. Pick "Sample vault (demo)"
3. Leave embedding backend on "Gemini (recommended)"
4. Click **Build / Rebuild Index**, then ask questions

Try:
- "What vector databases are good for RAG?"
- "How should I chunk my notes?"
- "Why did I build this project?"

To use your own vault: switch to "Upload my own .md files" in the sidebar,
upload notes, rebuild the index.

### Using the local embedding backend instead

Uncomment `sentence-transformers` in `requirements.txt`, reinstall, then
pick "Local (sentence-transformers)" in the sidebar. No API key needed for
indexing (still needed for generation). Downloads a ~90MB model on first
use and uses noticeably more RAM — fine locally, riskier on a free-tier
cloud host, which is why it's not the default.

## Deploying (for the live demo link)

Easiest path — **Streamlit Community Cloud** (free):
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   set `app.py` as the entry point.
3. Leave `sentence-transformers` commented out in `requirements.txt` for
   the deploy (keeps the app light and avoids the 1GB RAM limit).
4. In the app's "Secrets" settings, add:
   ```
   GEMINI_API_KEY = "your_key_here"
   ```
5. Deploy — you'll get a public `*.streamlit.app` URL. The key you set in
   Secrets pre-fills the sidebar field so evaluators don't need their own.

This project has zero compiled/Rust dependencies left, so it should install
cleanly regardless of which Python version Streamlit Cloud is currently
defaulting to.

(Hugging Face Spaces with the Streamlit SDK is a fine alternative.)

## Testing

```bash
pip install pytest
pytest tests/ -v
```

14 unit tests: markdown cleaning, the chunking algorithm, vault file
loading, and the vector store (add/query/persistence/clear), all
network-independent. The embedding/generation API calls themselves aren't
unit tested since they need a live key — verified manually against the
real Gemini API during development instead of assumed from documentation.

## Known limitations (MVP scope)

- Chunking is character-based, not token/semantic-aware.
- Index rebuilds are full rebuilds, not incremental.
- The vector store loads all embeddings into memory and does a linear scan
  per query (`O(n)`) — completely fine at note-collection scale (hundreds to
  low thousands of chunks), would need a proper ANN index (or reintroducing
  a vector DB, once the Python-version/dependency situation settles) at
  much larger scale.
- No conversation-aware retrieval (each question is retrieved independently).
- Gemini embedding backend calls the API once per chunk, not batched.

## What I'd add with more time

- Hybrid search (keyword + vector) for exact-term queries.
- Streaming responses instead of waiting for the full generation.
- Incremental re-indexing (only re-embed changed files).
- A small retrieval-quality eval set (sample Q&A pairs + expected source
  notes) to catch regressions when tuning chunk size/overlap.
