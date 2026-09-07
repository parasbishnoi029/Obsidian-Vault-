# Obsidian Vault Intelligence — User Manual

A Streamlit app that turns an Obsidian (Markdown) vault into a searchable,
question-answering system using retrieval-augmented generation (RAG). Every
answer is grounded in chunks retrieved from your own notes — the model isn't
asked to answer from general knowledge.

---

## 1. What you need before you start

- Python 3.10+
- A Gemini API key **only if** you plan to use the Gemini embedding backend
  (get one at https://aistudio.google.com/app/apikey). The Local backend
  (sentence-transformers) needs no key and runs entirely offline.
- Your Obsidian vault, or any folder of `.md` files.

---

## 2. Local setup

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

Configure your Gemini key (skip this if you'll only use the Local backend):

```bash
mkdir -p .streamlit
cp secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your real key
```

Run it:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 3. Using the app

The sidebar walks you through three steps — each one lights up gold once
it's satisfied:

**Step 1 — Select Vault**
Choose the bundled sample vault, or switch to "Upload Markdown files" and
drag in your own `.md` notes. The file count and size shown come straight
from what's on disk.

**Step 2 — Embedding Engine**
- *Gemini*: higher-quality cloud embeddings; requires `GEMINI_API_KEY`.
- *Local*: sentence-transformers running on your machine; no key, no
  network call, slightly lower embedding quality on niche vocabulary.

If you pick Gemini and no key is configured, the sidebar tells you exactly
that, right under this step.

**Step 3 — Knowledge Index**
Click **⚡ Build / Rebuild Index**. This reads every note, chunks it, embeds
each chunk, and stores it in the vector store. Rebuild whenever you change
vault, switch backend, or edit your notes — the app detects a mismatch
(via a content fingerprint) and marks the index stale automatically.

Once the index is ready, ask a question in the chat box at the bottom, or
use one of the three quick-start buttons ("Summarize my notes", "Find key
concepts", "Explain connections"). Each answer comes with an **Evidence
Used** expander showing exactly which chunks were retrieved and how
similar they were to your question — use this to sanity-check the answer
or track down the source note.

Use **Clear Conversation** in the sidebar to reset the chat without losing
your index.

---

## 4. Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (the `.gitignore` already keeps your local
   `secrets.toml` and uploaded files out of version control).
2. On https://share.streamlit.io, create a new app pointing at `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   ```
4. Deploy. The key is server-side only — visitors to your app never see it
   or enter one themselves.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Gemini API key is not configured" warning | No key in secrets/env | Add `GEMINI_API_KEY` under Settings → Secrets (cloud) or `.streamlit/secrets.toml` (local), or switch to Local backend |
| Build Index button is disabled | Vault or backend step not satisfied | Check the two step indicators above it in the sidebar |
| "No chunks were created" | Vault has no readable `.md` content | Confirm the uploaded/selected folder actually contains Markdown files |
| Chat answers "index is not ready" | You changed vault/backend since last build | Click Build / Rebuild Index again |
| Answer says no relevant evidence found | Question doesn't match anything in the vault | Rephrase, or check the vault actually covers that topic |

---

## 6. Notes on this build

- No API key is ever entered by end users — it's read server-side from
  `st.secrets` or the `GEMINI_API_KEY` environment variable only.
- Sidebar cards use `st.container(border=True)` rather than raw HTML divs,
  so styling genuinely wraps each step's controls.
- The hero's 3D orb animation and the ambient background both respect
  `prefers-reduced-motion`.
- This manual assumes the `rag/` package (ingest, retriever, generator,
  vectorstore) already exists in your project — it isn't part of this
  file set. Adjust `requirements.txt` to match what that package actually
  imports (e.g. swap `chromadb` out if you're using a different vector
  store).
