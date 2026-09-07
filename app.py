"""
app.py
Streamlit UI for the Obsidian Vault RAG Knowledge Assistant.

Flow:
1. User points the app at a folder of markdown notes (a sample vault ships
   with the repo, or they can upload their own .md files).
2. "Build / Rebuild Index" chunks + embeds every note into a persisted
   NumPy-based vector store (see rag/vectorstore.py).
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


def _index_chunk_count(persist_dir: str = CHROMA_DIR) -> int:
    try:
        return SimpleVectorStore(persist_dir, COLLECTION_NAME).count()
    except Exception:
        return 0


if "index_built" not in st.session_state:
    st.session_state["index_built"] = _index_exists_on_disk()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 Obsidian Vault RAG Knowledge Assistant")
st.caption("Ask questions about your notes. Answers are grounded only in what's actually in the vault.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    # Key stored in st.session_state (private to this browser session) rather
    # than os.environ, which is one shared process-wide dict on Streamlit
    # Cloud — writing a key there would leak it to every other visitor's
    # session. The box also isn't pre-filled with a stored value, since
    # Streamlit's password field can be un-hidden with its eye icon; a
    # pre-filled key would let anyone who opens the (public) app reveal it.
    key_input = st.text_input(
        "Gemini API key", type="password",
        value="",
        placeholder="Paste your key here (re-enter each session)",
        help="Free key at https://aistudio.google.com/apikey. Needed for both "
             "embeddings (Gemini backend) and answer generation. Not stored "
             "anywhere — you'll need to re-enter it if you reload the page."
    )
    if key_input:
        st.session_state["gemini_api_key"] = key_input
    api_key = st.session_state.get("gemini_api_key")
    st.caption("✅ Key set for this session" if api_key else "🔒 No key set yet")

    st.divider()

    st.subheader("1. Vault")
    source_choice = st.radio(
        "Use which notes?", ["Sample vault (demo)", "Upload my own .md files"],
        label_visibility="collapsed",
    )

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

    vault_ready = os.path.isdir(active_vault) and bool(os.listdir(active_vault))
    if not vault_ready:
        st.caption("⚠️ No notes in this vault yet.")

    st.subheader("2. Embedding backend")
    backend = st.radio(
        "How to embed notes",
        ["Gemini (recommended)", "Local (sentence-transformers)"],
        label_visibility="collapsed",
        help="Gemini: lightweight, works reliably on free-tier cloud deploys, needs the API key above. "
             "Local: runs offline, no API key needed, but downloads a ~90MB model and uses more RAM."
    )
    backend_key = "gemini" if backend.startswith("Gemini") else "local"

    st.subheader("3. Index")
    build_disabled = not vault_ready or (backend_key == "gemini" and not api_key)
    if st.button("🔨 Build / Rebuild Index", use_container_width=True, disabled=build_disabled):
        with st.spinner("Chunking + embedding notes... this can take a minute for larger vaults."):
            try:
                n = build_index(active_vault, backend=backend_key, api_key=api_key)
                if n == 0:
                    st.warning("No chunks were produced — check that the vault has readable .md files.")
                else:
                    st.success(f"Indexed {n} chunks from `{active_vault}` using the {backend_key} backend.")
                st.session_state["index_built"] = n > 0
                st.session_state["index_backend"] = backend_key
            except Exception as e:
                st.session_state["index_built"] = False
                st.error(f"Indexing failed: {e}")

    if not vault_ready:
        st.caption("Add or upload notes before building the index.")
    elif backend_key == "gemini" and not api_key:
        st.caption("Enter a Gemini API key above to build with this backend.")

    current_chunks = _index_chunk_count()
    if st.session_state.get("index_built") and current_chunks:
        built_with = st.session_state.get("index_backend", "unknown")
        st.caption(f"📚 Index ready — {current_chunks} chunks (built with **{built_with}**)")

    st.divider()
    top_k = st.slider("Chunks to retrieve", 2, 10, 5)

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------- chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your vault...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chunks = []
        if not st.session_state.get("index_built"):
            answer = ("No index has been built yet. Click **🔨 Build / Rebuild Index** "
                       "in the sidebar first, then ask again.")
            st.warning(answer)
        elif backend_key == "gemini" and not api_key:
            answer = "This index was built with Gemini embeddings — enter your API key in the sidebar to query it."
            st.warning(answer)
        else:
            with st.spinner("Searching vault + generating answer..."):
                try:
                    chunks = retrieve(question, top_k=top_k, api_key=api_key)
                    answer = generate_answer(question, chunks, api_key=api_key)
                except Exception as e:
                    chunks = []
                    answer = f"Something went wrong answering that: {e}"
                    st.error(answer)
            if chunks or "wrong answering" not in answer:
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
