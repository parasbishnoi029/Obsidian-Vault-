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

from rag.ingest import build_index
from rag.retriever import retrieve
from rag.generator import generate_answer

st.set_page_config(page_title="Obsidian Vault RAG Assistant", page_icon="🧠", layout="wide")

SAMPLE_VAULT = "data/sample_vault"
UPLOAD_VAULT = "data/uploaded_vault"

st.title("🧠 Obsidian Vault RAG Knowledge Assistant")
st.caption("Ask questions about your notes. Answers are grounded only in what's actually in the vault.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Gemini API key")
    key_input = st.text_input(
        "GEMINI_API_KEY", type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Free key at https://aistudio.google.com/apikey. Needed for both "
             "embeddings (Gemini backend) and answer generation."
    )
    if key_input:
        os.environ["GEMINI_API_KEY"] = key_input

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
        elif backend_key == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            st.error("Gemini backend selected but no API key is set above.")
        else:
            with st.spinner("Chunking + embedding notes..."):
                try:
                    n = build_index(active_vault, backend=backend_key)
                    st.success(f"Indexed {n} chunks from {active_vault} using the {backend_key} backend.")
                    st.session_state["index_built"] = True
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
        with st.spinner("Searching vault + generating answer..."):
            try:
                chunks = retrieve(question, top_k=top_k)
            except Exception as e:
                chunks = []
                st.error(f"Retrieval failed: {e}")
            answer = generate_answer(question, chunks)
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
