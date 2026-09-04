# Obsidian Vault RAG Knowledge Assistant

A retrieval-augmented generation (RAG) assistant that answers questions
grounded in your own Obsidian vault (a folder of markdown notes) — built as
a 1-week MVP for a Generative AI Developer Intern take-home.

## What it does

1. Point it at a folder of `.md` notes (a sample vault ships with the repo,
   or upload your own).
2. It chunks each note, embeds the chunks, and stores them in a vector
   database (Chroma).
3. Ask a question in the chat box. It retrieves the most relevant chunks
   and asks Gemini to answer **using only that context**, citing which
   note(s) the answer came from.
4. If no Gemini key is set, it still runs — it falls back to showing the
   raw retrieved passages, so the retrieval half of the pipeline is
   demonstrable even without an API key. (Generating an *index* still needs
   a key if using the Gemini embedding backend — see below.)

## Why this approach

I picked RAG over the scraping/agent options mainly for reliability under a
hard deadline — no live-site scraping to break against rate limits, no
multi-step agent loop to debug. Just a clean, testable pipeline: chunk →
embed → retrieve → ground → generate. That's the core skill the role is
evaluating, so I wanted each stage to be visible and independently testable
rather than hidden behind one black-box call.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| UI | Streamlit | fastest way to ship a usable chat UI + free one-click deploy |
| Embeddings | Gemini `gemini-embedding-001` (default), local `sentence-transformers` (optional) | Gemini backend has no heavy dependency, so it doesn't risk OOM-ing on free-tier cloud hosting. Local is there for fully offline dev. |
| Vector store | ChromaDB (persistent, local) | zero-config, no external service to provision |
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
                     chroma_store/      (persistent vector index)
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

(Hugging Face Spaces with the Streamlit SDK is a fine alternative.)

## Testing

```bash
pip install pytest
pytest tests/ -v
```

8 unit tests cover the network-independent parts of the pipeline: markdown
cleaning (wiki-links, embeds, tags), the chunking algorithm (short text,
long text with overlap, empty input), and vault file loading (including
nested folders). The embedding/generation calls themselves aren't unit
tested since they need a live API key — those were verified manually
against the real Gemini API during development (see commit history / dev
notes) rather than assumed to work from documentation alone.

## Known limitations (MVP scope)

- Chunking is character-based, not token/semantic-aware — fine for note-sized
  markdown, would want a smarter splitter for longer/denser documents.
- Index rebuilds are full rebuilds, not incremental.
- No conversation-aware retrieval (each question is retrieved independently,
  not re-written using chat history).
- Gemini embedding backend calls the API once per chunk (no batch embed
  endpoint used yet) — fine at demo scale, would batch for a larger vault.

## What I'd add with more time

- Hybrid search (keyword + vector) for exact-term queries.
- Streaming responses instead of waiting for the full generation.
- Incremental re-indexing (only re-embed changed files).
- A small retrieval-quality eval set (sample Q&A pairs + expected source
  notes) to catch regressions when tuning chunk size/overlap.
