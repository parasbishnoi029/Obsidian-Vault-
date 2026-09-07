"""
app.py
Streamlit UI for the Obsidian Vault RAG Knowledge Assistant.

Flow:
1. User points the app at a folder of markdown notes (a sample vault ships
   with the repo, or they can upload their own .md files).
2. "Build / Rebuild Index" chunks + embeds every note into Chroma.
3. User asks questions in a chat box; answers are grounded in retrieved
   chunks and cite the source note.
"""

import os
import streamlit as st

from rag.ingest import build_index, CHROMA_DIR, COLLECTION_NAME
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.vectorstore import SimpleVectorStore

st.set_page_config(page_title="Obsidian Vault RAG Assistant", page_icon="🧠", layout="wide")

SAMPLE_VAULT = "data/sample_vault"
UPLOAD_VAULT = "data/uploaded_vault"

def _index_exists_on_disk(persist_dir: str = CHROMA_DIR) -> bool:
    """True if a previously built index is persisted, so a page reload
    doesn't force the user to rebuild before asking a question."""
    try:
        return SimpleVectorStore(persist_dir, COLLECTION_NAME).count() > 0
    except Exception:
        return False


if "index_built" not in st.session_state:
    st.session_state["index_built"] = _index_exists_on_disk()

st.title("🧠 Obsidian Vault RAG Knowledge Assistant")
st.caption("Ask questions about your notes. Answers are grounded only in what's actually in the vault.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Gemini API key")
    # Kept in st.session_state (private to this browser session) rather than
    # os.environ, which is one shared process-wide dict on Streamlit Cloud —
    # writing a key there leaks it to every other visitor's session. The
    # widget also isn't pre-filled with a stored value, since Streamlit's
    # password field can be un-hidden with its eye icon; pre-filling it would
    # let anyone who opens the (public) app reveal whatever key is loaded.
    key_input = st.text_input(
        "GEMINI_API_KEY", type="password",
        value="",
        placeholder="Paste your key here (re-enter each session)",
        help="Free key at https://aistudio.google.com/apikey. Needed for both "
             "embeddings (Gemini backend) and answer generation. Not stored "
             "anywhere - you'll need to re-enter it if you reload the page."
    )
    if key_input:
        st.session_state["gemini_api_key"] = key_input

    api_key = st.session_state.get("gemini_api_key")
    if api_key:
        st.caption("✅ Key set for this session")

    st.header("2. Vault")
    source_choice = st.radio("Use which notes?", ["Sample vault (demo)", "Upload my own .md files"])

    if source_choice == "Upload my own .md files":
        uploaded = st.file_uploader("Upload markdown files", type=["md"], accept_multiple_files=True)
        if uploaded:
            os.makedirs(UPLOAD_VAULT, exist_ok=True)
            for f in uploaded:
                with open(os.path.join(UPLOAD_VAULT, f.name), "wb") as out:
                    out.write(f.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s).")
        active_vault = UPLOAD_VAULT
    else:
        active_vault = SAMPLE_VAULT

    st.header("3. Embedding backend")
    backend = st.radio(
        "How to embed notes",
        ["Gemini (recommended)", "Local (sentence-transformers)"],
        help="Gemini: lightweight, works reliably on free-tier cloud deploys, needs the API key above. "
             "Local: runs offline, no API key needed, but downloads a ~90MB model and uses more RAM."
    )
    backend_key = "gemini" if backend.startswith("Gemini") else "local"

    st.header("4. Index")
    if st.button("🔨 Build / Rebuild Index", use_container_width=True):
        if not os.path.isdir(active_vault) or not os.listdir(active_vault):
            st.error("No notes found in the selected vault.")
        elif backend_key == "gemini" and not api_key:
            st.error("Gemini backend selected but no API key is set above.")
        else:
            with st.spinner("Chunking + embedding notes..."):
                try:
                    n = build_index(active_vault, backend=backend_key, api_key=api_key)
                    st.success(f"Indexed {n} chunks from {active_vault} using the {backend_key} backend.")
                    st.session_state["index_built"] = True
                    st.session_state["index_backend"] = backend_key
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    top_k = st.slider("Chunks to retrieve", 2, 10, 5)

# ---------------------------------------------------------------- chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your vault...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.get("index_built"):
            answer = ("No index has been built yet. Click **🔨 Build / Rebuild Index** "
                       "in the sidebar first, then ask again.")
            chunks = []
            st.warning(answer)
        else:
            with st.spinner("Searching vault + generating answer..."):
                try:
                    chunks = retrieve(question, top_k=top_k, api_key=api_key)
                except Exception as e:
                    chunks = []
                    st.error(f"Retrieval failed: {e}")
                answer = generate_answer(question, chunks, api_key=api_key)
            st.markdown(answer)

        if chunks:
            with st.expander(f"📎 Sources used ({len(chunks)} chunks)"):
                for c in chunks:
                    st.markdown(f"**{c['source']}** (distance: {c['score']:.3f})")
                    st.code(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})

if not st.session_state.messages:
    st.info("👈 Set your Gemini key, build the index from the sidebar, then ask a question below. "
            "Try: *\"What are my notes on vector databases?\"* with the sample vault.")
