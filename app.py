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

API key: the Gemini key is configured server-side via Streamlit secrets
(GEMINI_API_KEY) rather than typed into the UI. A per-visitor text box on a
public app is the wrong shape for a secret — it either has to be re-entered
every session (annoying) or gets stored somewhere shared (risky). Setting it
once in the app's Secrets is the standard pattern for a single-owner
deployment like this one.

Visual design: themed around Obsidian's own graph-of-notes metaphor — the
sidebar setup steps are drawn as connected nodes that light up as they're
satisfied, with each step's control nested directly under its own node
instead of being separated from it. Chat turns render as bordered cards
with a colored role label. See the CSS block below for the token system.
"""

import os
import streamlit as st

from rag.ingest import build_index, CHROMA_DIR, COLLECTION_NAME
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.vectorstore import SimpleVectorStore

st.set_page_config(page_title="Obsidian Vault RAG Assistant", page_icon="◈", layout="wide")

SAMPLE_VAULT = "data/sample_vault"
UPLOAD_VAULT = "data/uploaded_vault"


def _get_api_key():
    """Server-side only: Streamlit secrets first, then env var. Never a
    UI text box — see module docstring for why."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


# ---------------------------------------------------------------- styling
def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #12101c;
        --surface: #1a1830;
        --surface-2: #221f3d;
        --border: #322d54;
        --text: #e9e6f5;
        --text-dim: #a9a3c4;
        --gold: #d9a94e;
        --gold-dim: #8a713a;
        --violet: #6d6aff;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .brand-title { font-family: 'Space Grotesk', sans-serif; }

    [data-testid="stAppViewContainer"] {
        background: var(--bg);
        position: relative;
        overflow: hidden;
    }

    /* one deliberate ambient moment: a slow drifting constellation, not
       motion on every element */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: -10%;
        background-image:
            radial-gradient(2px 2px at 20% 30%, rgba(217,169,78,0.35) 0, transparent 60%),
            radial-gradient(2px 2px at 70% 20%, rgba(109,106,255,0.30) 0, transparent 60%),
            radial-gradient(1.5px 1.5px at 40% 70%, rgba(217,169,78,0.25) 0, transparent 60%),
            radial-gradient(1.5px 1.5px at 85% 65%, rgba(109,106,255,0.25) 0, transparent 60%),
            radial-gradient(1.5px 1.5px at 55% 45%, rgba(217,169,78,0.2) 0, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }
    @media (prefers-reduced-motion: no-preference) {
        [data-testid="stAppViewContainer"]::before {
            animation: drift 60s ease-in-out infinite alternate;
        }
    }
    @keyframes drift {
        from { transform: translate(0, 0); }
        to   { transform: translate(-2%, 2%); }
    }

    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text); }

    /* ---------- brand header ---------- */
    .brand-row { display: flex; align-items: center; gap: 14px; margin-bottom: 2px; }
    .brand-title { font-size: 2.1rem; font-weight: 700; color: var(--text); margin: 0; letter-spacing: -0.01em; }
    .brand-sub { color: var(--text-dim); font-size: 0.95rem; margin-top: 2px; }

    /* ---------- stepper: one continuous line behind interleaved nodes+controls ---------- */
    .step-flow { position: relative; padding-left: 6px; }
    .step-flow::before {
        content: "";
        position: absolute;
        left: 15px; top: 10px; bottom: 26px;
        width: 2px;
        background: var(--border);
        z-index: 0;
    }
    .step-node {
        position: relative; z-index: 1;
        display: flex; align-items: flex-start; gap: 10px;
        margin: 14px 0 6px 0;
    }
    .step-orb {
        flex: 0 0 auto;
        width: 20px; height: 20px; border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #3a3660, var(--surface-2));
        border: 2px solid var(--border);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.65rem; color: var(--text-dim);
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.4);
        transition: all 0.35s ease;
    }
    .step-node.done .step-orb {
        background: radial-gradient(circle at 35% 30%, #f0cf85, var(--gold));
        border-color: var(--gold);
        color: #1a1408;
        box-shadow: 0 0 10px rgba(217,169,78,0.55), inset 0 1px 2px rgba(255,255,255,0.4);
    }
    .step-node.active .step-orb {
        border-color: var(--violet);
        box-shadow: 0 0 8px rgba(109,106,255,0.5);
    }
    .step-label { font-weight: 600; font-size: 0.92rem; color: var(--text); line-height: 1.1; }
    .step-hint { font-size: 0.78rem; color: var(--text-dim); margin-top: 1px; }
    .step-body { margin-left: 30px; margin-bottom: 4px; }

    /* ---------- buttons ---------- */
    div[data-testid="stSidebar"] button {
        background: linear-gradient(180deg, #2a2650, var(--surface-2));
        border: 1px solid var(--border);
        color: var(--text);
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stSidebar"] button:not(:disabled):hover {
        transform: translateY(-2px) perspective(300px) rotateX(3deg);
        border-color: var(--gold);
        box-shadow: 0 6px 16px rgba(217,169,78,0.25);
    }
    div[data-testid="stSidebar"] button:disabled { opacity: 0.45; }

    /* ---------- chat message role label ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); }
    .msg-role {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
        margin-bottom: 6px; text-transform: uppercase;
    }

    .source-slip {
        font-family: 'Space Grotesk', monospace;
        font-size: 0.82rem;
        border-left: 2px solid var(--gold-dim);
        padding: 4px 10px;
        margin-bottom: 6px;
        color: var(--text-dim);
    }
    .source-slip b { color: var(--text); }
    </style>
    """, unsafe_allow_html=True)


def _logo_svg() -> str:
    return """
    <svg width="38" height="38" viewBox="0 0 38 38" xmlns="http://www.w3.org/2000/svg">
        <line x1="10" y1="10" x2="28" y2="14" stroke="#6d6aff" stroke-width="1.4" opacity="0.7"/>
        <line x1="10" y1="10" x2="15" y2="28" stroke="#6d6aff" stroke-width="1.4" opacity="0.7"/>
        <line x1="28" y1="14" x2="15" y2="28" stroke="#d9a94e" stroke-width="1.4" opacity="0.7"/>
        <circle cx="10" cy="10" r="4" fill="#6d6aff"/>
        <circle cx="28" cy="14" r="3.4" fill="#8f8dff"/>
        <circle cx="15" cy="28" r="4.6" fill="#d9a94e"/>
    </svg>
    """


def _step_node(done: bool, active: bool, orb_text: str, label: str, hint: str):
    state = "done" if done else ("active" if active else "")
    st.markdown(
        f'<div class="step-node {state}"><div class="step-orb">{orb_text}</div>'
        f'<div><div class="step-label">{label}</div><div class="step-hint">{hint}</div></div></div>',
        unsafe_allow_html=True,
    )


def _render_message(role: str, content: str):
    is_assistant = role == "assistant"
    label = "Assistant" if is_assistant else "You"
    accent = "var(--gold)" if is_assistant else "var(--violet)"
    with st.container(border=True):
        st.markdown(
            f'<div class="msg-role" style="color:{accent}">{label}</div>',
            unsafe_allow_html=True,
        )
        # plain st.markdown (not escaped) so the generator's bullet points
        # and [source: filename] citations render instead of showing as
        # literal asterisks/brackets
        st.markdown(content)


# ---------------------------------------------------------------- state helpers
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

api_key = _get_api_key()

_inject_css()

st.markdown(
    f'<div class="brand-row">{_logo_svg()}<div><p class="brand-title">Obsidian Vault RAG Assistant</p></div></div>'
    f'<p class="brand-sub">Ask questions about your notes. Answers are grounded only in what\'s actually in the vault.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="step-flow">', unsafe_allow_html=True)

    # ---- step 1: vault ----
    _step_node(True, False, "1", "Vault", "Choose your notes")
    st.markdown('<div class="step-body">', unsafe_allow_html=True)
    source_choice = st.radio(
        "Vault", ["Sample vault (demo)", "Upload my own .md files"],
        label_visibility="collapsed",
    )
    if source_choice == "Upload my own .md files":
        uploaded = st.file_uploader("Upload markdown files", type=["md"], accept_multiple_files=True,
                                     label_visibility="collapsed")
        if uploaded:
            os.makedirs(UPLOAD_VAULT, exist_ok=True)
            for f in uploaded:
                with open(os.path.join(UPLOAD_VAULT, f.name), "wb") as out:
                    out.write(f.getbuffer())
            st.caption(f"Saved {len(uploaded)} file(s).")
        active_vault = UPLOAD_VAULT
    else:
        active_vault = SAMPLE_VAULT
    vault_ready = os.path.isdir(active_vault) and bool(os.listdir(active_vault))
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- step 2: embedding backend ----
    _step_node(vault_ready, not vault_ready, "2", "Embedding backend",
               "Gemini needs a key configured on the server; Local needs none")
    st.markdown('<div class="step-body">', unsafe_allow_html=True)
    backend = st.radio(
        "Embedding backend",
        ["Gemini (recommended)", "Local (sentence-transformers)"],
        label_visibility="collapsed",
    )
    backend_key = "gemini" if backend.startswith("Gemini") else "local"
    backend_ready = vault_ready and (backend_key == "local" or bool(api_key))
    if backend_key == "gemini" and not api_key:
        st.caption("⚠️ No GEMINI_API_KEY configured for this app — set it in "
                   "Settings → Secrets, or switch to Local.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- step 3: index ----
    current_chunks = _index_chunk_count()
    index_ready = st.session_state.get("index_built") and current_chunks > 0
    _step_node(index_ready, backend_ready and not index_ready, "3", "Index",
               f"{current_chunks} chunks ready" if index_ready else "Not built yet")
    st.markdown('<div class="step-body">', unsafe_allow_html=True)
    if st.button("Build / Rebuild Index", use_container_width=True, disabled=not backend_ready):
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
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close .step-flow

    st.divider()
    top_k = st.slider("Chunks to retrieve", 2, 10, 5)

    if st.session_state.messages:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------- chat
for msg in st.session_state.messages:
    _render_message(msg["role"], msg["content"])

question = st.chat_input("Ask something about your vault...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    _render_message("user", question)

    chunks = []
    if not st.session_state.get("index_built"):
        answer = "No index has been built yet. Build it from the sidebar first, then ask again."
    elif backend_key == "gemini" and not api_key:
        answer = ("This index needs Gemini embeddings, but no GEMINI_API_KEY is configured for "
                  "this app right now. Ask the app owner to set one, or rebuild using the Local backend.")
    else:
        with st.spinner("Searching vault + generating answer..."):
            try:
                chunks = retrieve(question, top_k=top_k, api_key=api_key)
                answer = generate_answer(question, chunks, api_key=api_key)
            except Exception as e:
                chunks = []
                answer = f"Something went wrong answering that: {e}"

    _render_message("assistant", answer)

    if chunks:
        with st.expander(f"Sources used ({len(chunks)} chunks)"):
            for c in chunks:
                st.markdown(
                    f'<div class="source-slip"><b>{c["source"]}</b> · distance {c["score"]:.3f}</div>',
                    unsafe_allow_html=True,
                )
                st.code(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})

if not st.session_state.messages:
    st.info(
        "Build the index from the sidebar, then ask a question below. "
        "Try: *\"What are my notes on vector databases?\"* with the sample vault."
    )
