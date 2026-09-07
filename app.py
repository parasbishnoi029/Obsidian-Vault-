

from __future__ import annotations

import hashlib
import html
import os
import shutil
from pathlib import Path
from typing import Any

import streamlit as st

from rag.ingest import build_index, CHROMA_DIR, COLLECTION_NAME
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.vectorstore import SimpleVectorStore


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Obsidian Vault Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONSTANTS
# ============================================================================

SAMPLE_VAULT = Path("data/sample_vault")
UPLOAD_VAULT = Path("data/uploaded_vault")

DEFAULT_TOP_K = 5
MIN_TOP_K = 2
MAX_TOP_K = 10

# Adjust this based on how your vector store represents similarity/distance.
# If lower scores mean better matches, this is a maximum acceptable distance.
DEFAULT_MAX_DISTANCE = 1.50


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_api_key() -> str | None:
    """
    Read the Gemini key from server-side configuration.

    Priority:
    1. Streamlit Secrets
    2. Environment variable

    Never request the key through the public UI.
    """

    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.environ.get("GEMINI_API_KEY")

    return key.strip() if key else None


# ============================================================================
# VAULT UTILITIES
# ============================================================================

def get_markdown_files(vault_path: Path) -> list[Path]:
    """Return all Markdown files recursively."""

    if not vault_path.exists():
        return []

    return sorted(
        [
            path
            for path in vault_path.rglob("*.md")
            if path.is_file()
        ]
    )


def vault_fingerprint(vault_path: Path) -> str:
    """
    Create a stable fingerprint for the vault.

    The fingerprint changes when:
    - files are added
    - files are removed
    - file contents change
    """

    hasher = hashlib.sha256()

    for file_path in get_markdown_files(vault_path):

        relative_path = str(file_path.relative_to(vault_path))

        hasher.update(relative_path.encode("utf-8"))

        try:
            hasher.update(file_path.read_bytes())
        except OSError:
            # Include unreadable files in a deterministic way.
            hasher.update(b"UNREADABLE")

    return hasher.hexdigest()[:16]


def vault_stats(vault_path: Path) -> dict[str, Any]:
    """Calculate basic vault statistics."""

    files = get_markdown_files(vault_path)

    total_bytes = 0

    for file in files:
        try:
            total_bytes += file.stat().st_size
        except OSError:
            continue

    return {
        "files": len(files),
        "bytes": total_bytes,
        "fingerprint": vault_fingerprint(vault_path) if files else None,
    }


def human_file_size(size: int) -> str:

    units = ["B", "KB", "MB", "GB"]

    value = float(size)

    for unit in units:

        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"

        value /= 1024

    return f"{value:.1f} GB"


def save_uploaded_files(uploaded_files: list[Any]) -> int:
    """
    Replace the previous uploaded vault with the current upload.

    This avoids a serious bug where old uploaded notes remain on disk
    and accidentally become part of future indexes.
    """

    if UPLOAD_VAULT.exists():
        shutil.rmtree(UPLOAD_VAULT)

    UPLOAD_VAULT.mkdir(parents=True, exist_ok=True)

    saved = 0

    for uploaded_file in uploaded_files:

        # Prevent directory traversal.
        safe_name = Path(uploaded_file.name).name

        if not safe_name.lower().endswith(".md"):
            continue

        target = UPLOAD_VAULT / safe_name

        target.write_bytes(uploaded_file.getbuffer())

        saved += 1

    return saved


# ============================================================================
# INDEX UTILITIES
# ============================================================================

def vector_store() -> SimpleVectorStore:
    return SimpleVectorStore(
        CHROMA_DIR,
        COLLECTION_NAME,
    )


def index_chunk_count() -> int:

    try:
        return vector_store().count()
    except Exception:
        return 0


def index_exists() -> bool:
    return index_chunk_count() > 0


def current_index_is_valid(
    vault_path: Path,
    backend: str,
) -> bool:
    """
    Check whether the current persisted index matches the active vault
    and embedding backend.
    """

    metadata = st.session_state.get("index_metadata")

    if not metadata:
        return False

    if not index_exists():
        return False

    current_fingerprint = vault_fingerprint(vault_path)

    return (
        metadata.get("vault_fingerprint") == current_fingerprint
        and metadata.get("backend") == backend
    )


# ============================================================================
# SESSION STATE
# ============================================================================

def initialize_session_state() -> None:

    defaults = {
        "messages": [],
        "index_metadata": None,
        "uploaded_signature": None,
        "last_build_error": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================================
# UI STYLING
# ============================================================================

def inject_css() -> None:

    st.markdown(
        """
        <style>

        @import url(
            'https://fonts.googleapis.com/css2?family=
            Inter:wght@400;500;600;700&
            family=Space+Grotesk:wght@500;600;700&
            display=swap'
        );

        :root {

            --bg: #0e0d16;
            --surface: #171526;
            --surface-2: #211e35;

            --border: #373154;
            --border-soft: rgba(255,255,255,0.06);

            --text: #f1eff8;
            --text-muted: #a6a1bd;

            --violet: #827dff;
            --violet-glow: rgba(130,125,255,0.28);

            --gold: #e1b55c;
            --gold-glow: rgba(225,181,92,0.25);

            --green: #65d69c;
            --red: #ff7777;
        }


        html,
        body,
        [class*="css"] {

            font-family: "Inter", sans-serif;

        }


        h1,
        h2,
        h3,
        .brand-title {

            font-family: "Space Grotesk", sans-serif;

        }


        [data-testid="stAppViewContainer"] {

            background:
                radial-gradient(
                    circle at 15% 10%,
                    rgba(109,106,255,0.08),
                    transparent 28%
                ),

                radial-gradient(
                    circle at 85% 20%,
                    rgba(217,169,78,0.06),
                    transparent 24%
                ),

                var(--bg);

        }


        [data-testid="stSidebar"] {

            background: var(--surface);

            border-right:
                1px solid var(--border-soft);

        }


        /* ------------------------------------------------------------
           BRAND
        ------------------------------------------------------------ */

        .brand {

            padding:
                0.6rem 0 1.6rem 0;

        }


        .brand-row {

            display: flex;

            align-items: center;

            gap: 14px;

        }


        .brand-icon {

            width: 44px;
            height: 44px;

            display: flex;

            align-items: center;

            justify-content: center;

            border-radius: 14px;

            background:
                linear-gradient(
                    145deg,
                    rgba(130,125,255,0.22),
                    rgba(225,181,92,0.12)
                );

            border:
                1px solid rgba(255,255,255,0.09);

            font-size: 1.5rem;

            box-shadow:
                0 12px 30px rgba(0,0,0,0.25);

        }


        .brand-title {

            margin: 0;

            font-size: 2rem;

            font-weight: 700;

            letter-spacing: -0.04em;

            color: var(--text);

        }


        .brand-subtitle {

            margin-top: 0.4rem;

            color: var(--text-muted);

            font-size: 0.95rem;

            max-width: 760px;

        }


        /* ------------------------------------------------------------
           STEP CARDS
        ------------------------------------------------------------ */

        .step-card {

            margin: 0.8rem 0;

            padding: 0.85rem;

            border-radius: 14px;

            background:
                rgba(255,255,255,0.025);

            border:
                1px solid var(--border-soft);

        }


        .step-header {

            display: flex;

            align-items: center;

            gap: 10px;

            margin-bottom: 0.55rem;

        }


        .step-number {

            width: 26px;

            height: 26px;

            border-radius: 50%;

            display: flex;

            align-items: center;

            justify-content: center;

            font-size: 0.78rem;

            font-weight: 700;

            background: var(--surface-2);

            border: 1px solid var(--border);

            color: var(--text-muted);

        }


        .step-number.done {

            color: #14100a;

            background: var(--gold);

            border-color: var(--gold);

            box-shadow:
                0 0 16px var(--gold-glow);

        }


        .step-number.active {

            border-color: var(--violet);

            box-shadow:
                0 0 14px var(--violet-glow);

        }


        .step-title {

            font-weight: 700;

            color: var(--text);

            font-size: 0.9rem;

        }


        .step-description {

            font-size: 0.75rem;

            color: var(--text-muted);

        }


        /* ------------------------------------------------------------
           CHAT
        ------------------------------------------------------------ */

        .message-role {

            font-size: 0.7rem;

            font-weight: 700;

            letter-spacing: 0.08em;

            text-transform: uppercase;

            margin-bottom: 0.45rem;

        }


        .message-role.user {

            color: var(--violet);

        }


        .message-role.assistant {

            color: var(--gold);

        }


        .source-card {

            padding: 0.75rem;

            margin-bottom: 0.7rem;

            border-radius: 10px;

            background:
                rgba(255,255,255,0.025);

            border-left:
                3px solid var(--gold);

        }


        .source-name {

            font-family:
                "Space Grotesk",
                monospace;

            font-weight: 600;

            color: var(--text);

        }


        .source-score {

            font-size: 0.78rem;

            color: var(--text-muted);

        }


        /* ------------------------------------------------------------
           METRICS
        ------------------------------------------------------------ */

        [data-testid="stMetric"] {

            padding: 0.7rem;

            border-radius: 12px;

            background:
                rgba(255,255,255,0.025);

            border:
                1px solid var(--border-soft);

        }

        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_brand() -> None:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-row">

                <div class="brand-icon">◈</div>

                <div>
                    <div class="brand-title">
                        Obsidian Vault Intelligence
                    </div>
                </div>

            </div>

            <div class="brand-subtitle">

                Ask questions about your Markdown knowledge base.
                Answers are generated from retrieved evidence, not guesswork.

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step(
    number: int,
    title: str,
    description: str,
    *,
    done: bool = False,
    active: bool = False,
) -> None:

    state = "done" if done else "active" if active else ""

    st.markdown(
        f"""
        <div class="step-header">

            <div class="step-number {state}">
                {number}
            </div>

            <div>

                <div class="step-title">
                    {html.escape(title)}
                </div>

                <div class="step-description">
                    {html.escape(description)}
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(role: str, content: str) -> None:

    label = "Assistant" if role == "assistant" else "You"

    with st.container(border=True):

        st.markdown(
            f'<div class="message-role {role}">{label}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(content)


def render_index_status(
    vault_path: Path,
    backend: str,
    index_valid: bool,
) -> None:

    stats = vault_stats(vault_path)

    chunks = index_chunk_count()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Notes",
        stats["files"],
    )

    col2.metric(
        "Chunks",
        chunks,
    )

    col3.metric(
        "Engine",
        backend.title(),
    )

    if index_valid:

        st.success("Knowledge index is ready and matches the active vault.")

    elif chunks > 0:

        st.warning(
            "An index exists, but it does not match the current vault "
            "or embedding backend. Rebuild before querying."
        )

    else:

        st.info(
            "No knowledge index exists yet."
        )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

api_key = get_api_key()

render_brand()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown("## Vault Control")

    # ----------------------------------------------------------------------
    # STEP 1 — VAULT
    # ----------------------------------------------------------------------

    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    source_choice = st.radio(
        "Vault source",
        [
            "Sample vault",
            "Upload Markdown files",
        ],
        label_visibility="collapsed",
    )

    if source_choice == "Sample vault":

        active_vault = SAMPLE_VAULT

        vault_ready = len(get_markdown_files(active_vault)) > 0

        render_step(
            1,
            "Select Vault",
            "Using the repository sample knowledge vault",
            done=vault_ready,
            active=not vault_ready,
        )

    else:

        uploaded_files = st.file_uploader(
            "Upload Markdown files",
            type=["md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:

            upload_signature = hashlib.sha256(
                "".join(
                    f"{file.name}:{file.size}"
                    for file in uploaded_files
                ).encode("utf-8")
            ).hexdigest()

            if (
                upload_signature
                != st.session_state.get("uploaded_signature")
            ):

                saved = save_uploaded_files(uploaded_files)

                st.session_state["uploaded_signature"] = upload_signature

                st.toast(
                    f"Loaded {saved} Markdown file(s).",
                    icon="📚",
                )

        active_vault = UPLOAD_VAULT

        vault_ready = (
            len(get_markdown_files(active_vault)) > 0
        )

        render_step(
            1,
            "Select Vault",
            "Upload one or more Markdown notes",
            done=vault_ready,
            active=not vault_ready,
        )

        if vault_ready:

            stats = vault_stats(active_vault)

            st.caption(
                f"{stats['files']} note(s) • "
                f"{human_file_size(stats['bytes'])}"
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # STEP 2 — BACKEND
    # ----------------------------------------------------------------------

    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    backend_label = st.radio(
        "Embedding backend",
        [
            "Gemini",
            "Local sentence-transformers",
        ],
        label_visibility="collapsed",
    )

    backend = (
        "gemini"
        if backend_label == "Gemini"
        else "local"
    )

    backend_ready = (
        vault_ready
        and (
            backend == "local"
            or bool(api_key)
        )
    )

    render_step(
        2,
        "Embedding Engine",
        (
            "Gemini embeddings"
            if backend == "gemini"
            else "Local embeddings — no API key required"
        ),
        done=backend_ready,
        active=vault_ready and not backend_ready,
    )

    if backend == "gemini" and not api_key:

        st.warning(
            "Gemini is selected, but GEMINI_API_KEY is not configured "
            "on the server. Switch to Local or configure Streamlit Secrets."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # STEP 3 — INDEX
    # ----------------------------------------------------------------------

    index_valid = current_index_is_valid(
        active_vault,
        backend,
    )

    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    render_step(
        3,
        "Build Knowledge Index",
        (
            "Ready to query"
            if index_valid
            else "Chunk and embed the active vault"
        ),
        done=index_valid,
        active=backend_ready and not index_valid,
    )

    build_disabled = not backend_ready

    if st.button(
        "Build / Rebuild Index",
        use_container_width=True,
        disabled=build_disabled,
    ):

        with st.spinner(
            "Reading notes, chunking documents and building embeddings..."
        ):

            try:

                chunk_count = build_index(
                    str(active_vault),
                    backend=backend,
                    api_key=api_key,
                )

                if chunk_count <= 0:

                    st.session_state["index_metadata"] = None

                    st.error(
                        "No chunks were created. "
                        "Check that the vault contains readable Markdown files."
                    )

                else:

                    stats = vault_stats(active_vault)

                    st.session_state["index_metadata"] = {
                        "vault_fingerprint": stats["fingerprint"],
                        "backend": backend,
                        "vault_path": str(active_vault),
                        "chunk_count": chunk_count,
                    }

                    # Prevent old answers from being confused with a new index.
                    st.session_state["messages"] = []

                    st.success(
                        f"Knowledge index built successfully: "
                        f"{chunk_count} chunks."
                    )

                    st.toast(
                        "Index is ready. Conversation was reset.",
                        icon="✓",
                    )

            except Exception:

                st.session_state["index_metadata"] = None

                st.session_state["last_build_error"] = True

                st.error(
                    "Indexing failed. Check the server logs and verify "
                    "your embedding configuration."
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # STATUS
    # ----------------------------------------------------------------------

    st.divider()

    st.markdown("### Index Status")

    render_index_status(
        active_vault,
        backend,
        index_valid,
    )

    # ----------------------------------------------------------------------
    # RETRIEVAL SETTINGS
    # ----------------------------------------------------------------------

    st.divider()

    st.markdown("### Retrieval")

    top_k = st.slider(
        "Chunks to retrieve",
        MIN_TOP_K,
        MAX_TOP_K,
        DEFAULT_TOP_K,
    )

    max_distance = st.slider(
        "Maximum retrieval distance",
        min_value=0.10,
        max_value=3.00,
        value=DEFAULT_MAX_DISTANCE,
        step=0.05,
        help=(
            "Lower is stricter. Results with worse scores are discarded. "
            "Tune this based on your vector store's distance metric."
        ),
    )

    # ----------------------------------------------------------------------
    # CONVERSATION CONTROL
    # ----------------------------------------------------------------------

    if st.session_state["messages"]:

        st.divider()

        if st.button(
            "Clear Conversation",
            use_container_width=True,
        ):

            st.session_state["messages"] = []

            st.rerun()


# ============================================================================
# CHAT HISTORY
# ============================================================================

for message in st.session_state["messages"]:

    render_message(
        message["role"],
        message["content"],
    )


# ============================================================================
# EMPTY STATE
# ============================================================================

if not st.session_state["messages"]:

    if index_valid:

        st.info(
            "The knowledge index is ready. "
            "Ask a question about your notes below."
        )

    else:

        st.info(
            "Select a vault, configure an embedding engine, "
            "and build the index."
        )


# ============================================================================
# CHAT INPUT
# ============================================================================

question = st.chat_input(
    "Ask something about your knowledge vault..."
)


if question:

    # ----------------------------------------------------------------------
    # USER MESSAGE
    # ----------------------------------------------------------------------

    st.session_state["messages"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    render_message(
        "user",
        question,
    )

    chunks: list[dict[str, Any]] = []

    # ----------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------

    if not index_valid:

        answer = (
            "The active knowledge index is missing or does not match "
            "the currently selected vault/backend. "
            "Rebuild the index from the sidebar first."
        )

    elif backend == "gemini" and not api_key:

        answer = (
            "This configuration requires Gemini, but the server does not "
            "currently have a GEMINI_API_KEY configured."
        )

    else:

        # ------------------------------------------------------------------
        # RETRIEVAL + GENERATION
        # ------------------------------------------------------------------

        try:

            with st.spinner(
                "Searching the vault and checking evidence..."
            ):

                chunks = retrieve(
                    question,
                    top_k=top_k,
                    api_key=api_key,
                )

            # --------------------------------------------------------------
            # RETRIEVAL CONFIDENCE FILTER
            # --------------------------------------------------------------

            chunks = [
                chunk
                for chunk in chunks
                if float(chunk.get("score", float("inf")))
                <= max_distance
            ]

            if not chunks:

                answer = (
                    "I couldn't find sufficiently relevant evidence in the "
                    "current vault to answer that reliably."
                )

            else:

                with st.spinner(
                    "Generating a grounded answer..."
                ):

                    answer = generate_answer(
                        question,
                        chunks,
                        api_key=api_key,
                    )

        except Exception:

            chunks = []

            answer = (
                "The request could not be completed. "
                "Check the server logs, embedding backend, and API configuration."
            )

    # ----------------------------------------------------------------------
    # ASSISTANT MESSAGE
    # ----------------------------------------------------------------------

    render_message(
        "assistant",
        answer,
    )

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    # ----------------------------------------------------------------------
    # EVIDENCE PANEL
    # ----------------------------------------------------------------------

    if chunks:

        with st.expander(
            f"Evidence used — {len(chunks)} retrieved chunk(s)",
            expanded=False,
        ):

            for index, chunk in enumerate(chunks, start=1):

                source = html.escape(
                    str(chunk.get("source", "Unknown source"))
                )

                score = float(
                    chunk.get("score", 0)
                )

                text = str(
                    chunk.get("text", "")
                )

                st.markdown(
                    f"""
                    <div class="source-card">

                        <div class="source-name">
                            {index}. {source}
                        </div>

                        <div class="source-score">
                            Retrieval distance: {score:.3f}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.code(
                    text[:500]
                    + (
                        "..."
                        if len(text) > 500
                        else ""
                    ),
                    language="markdown",
                )

