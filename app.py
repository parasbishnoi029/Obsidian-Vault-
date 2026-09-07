import os
import shutil
import hashlib
import html
from pathlib import Path

import streamlit as st

from rag.ingest import build_index, CHROMA_DIR, COLLECTION_NAME
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.vectorstore import SimpleVectorStore


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Obsidian Vault Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONFIG
# =============================================================================

SAMPLE_VAULT = Path("data/sample_vault")
UPLOAD_VAULT = Path("data/uploaded_vault")

MIN_TOP_K = 2
MAX_TOP_K = 10
DEFAULT_TOP_K = 5


# =============================================================================
# HELPERS
# =============================================================================

def clean_html(content: str) -> str:
    """
    Critical helper.

    textwrap.dedent only strips whitespace common to every line, which
    is not enough for nested HTML templates (a div inside a div inside
    an f-string) — inner lines are still left with 4+ leading spaces,
    and Markdown treats 4+ leading spaces as a literal code block. That
    silently turns real HTML into visible text like "<div class=...>".

    Stripping every line individually guarantees no line ever hits that
    4-space threshold, regardless of how deeply the template is nested.
    Safe here because we only ever emit block-level tags (div/span),
    where inter-tag whitespace has no visual effect.
    """
    lines = content.strip("\n").split("\n")
    return "\n".join(line.strip() for line in lines)


def render_html(content: str):
    st.markdown(
        clean_html(content),
        unsafe_allow_html=True,
    )


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.environ.get("GEMINI_API_KEY")
    return key.strip() if key else None


def get_markdown_files(vault_path: Path):
    if not vault_path.exists():
        return []

    return sorted(
        [
            file
            for file in vault_path.rglob("*.md")
            if file.is_file()
        ]
    )


def get_vault_fingerprint(vault_path: Path):

    files = get_markdown_files(vault_path)

    if not files:
        return None

    hasher = hashlib.sha256()

    for file in files:

        try:
            relative = str(file.relative_to(vault_path))
            hasher.update(relative.encode())

            hasher.update(file.read_bytes())

        except Exception:
            continue

    return hasher.hexdigest()[:16]


def get_vault_stats(vault_path: Path):

    files = get_markdown_files(vault_path)

    total_size = 0

    for file in files:
        try:
            total_size += file.stat().st_size
        except Exception:
            pass

    return {
        "files": len(files),
        "size": total_size,
        "fingerprint": get_vault_fingerprint(vault_path),
    }


def format_size(size):

    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"

    return f"{size / (1024 * 1024):.2f} MB"


def get_chunk_count():

    try:
        store = SimpleVectorStore(
            CHROMA_DIR,
            COLLECTION_NAME,
        )

        return store.count()

    except Exception:
        return 0


def clear_uploaded_vault():

    if UPLOAD_VAULT.exists():
        shutil.rmtree(UPLOAD_VAULT)

    UPLOAD_VAULT.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_uploaded_files(files):

    clear_uploaded_vault()

    saved = 0

    for file in files:

        safe_name = Path(file.name).name

        if not safe_name.lower().endswith(".md"):
            continue

        destination = UPLOAD_VAULT / safe_name

        with open(destination, "wb") as output:
            output.write(file.getbuffer())

        saved += 1

    return saved


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_metadata" not in st.session_state:
    st.session_state.index_metadata = None

if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None


# =============================================================================
# CSS
# =============================================================================

def inject_css():

    render_html(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');


        :root {

            --bg: #090a12;
            --surface: rgba(20, 22, 35, 0.82);
            --surface-2: rgba(31, 34, 53, 0.85);
            --surface-3: rgba(44, 47, 70, 0.85);
            --border: rgba(255,255,255,0.08);
            --border-bright: rgba(130,125,255,0.45);
            --text: #f4f2ff;
            --text-muted: #9d9bb4;
            --violet: #827dff;
            --violet-2: #a19eff;
            --gold: #f0b95d;
            --cyan: #55d6ff;
            --green: #63e6a2;
            --red: #ff6b7a;

        }


        [data-testid="stAppViewContainer"] {

            background:
                radial-gradient(circle at 10% 15%, rgba(130,125,255,0.12), transparent 30%),
                radial-gradient(circle at 85% 20%, rgba(240,185,93,0.08), transparent 25%),
                radial-gradient(circle at 50% 90%, rgba(85,214,255,0.06), transparent 30%),
                var(--bg);
            overflow: hidden;

        }


        [data-testid="stAppViewContainer"]::before {

            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.65;
            background-image:
                radial-gradient(circle, rgba(130,125,255,0.65) 1px, transparent 1.5px),
                radial-gradient(circle, rgba(240,185,93,0.5) 1px, transparent 1.5px);
            background-size: 140px 140px, 210px 210px;
            background-position: 0 0, 60px 90px;
            mask-image: radial-gradient(ellipse at center, black, transparent 78%);
            animation: backgroundFloat 25s linear infinite;

        }


        @media (prefers-reduced-motion: reduce) {

            [data-testid="stAppViewContainer"]::before {
                animation: none;
            }

            .hero {
                animation: none;
            }

            .orb-3d, .orb-3d .ring {
                animation: none !important;
            }

        }


        /* ==========================================================
           3D KNOWLEDGE ORB — genuine 3D transform, not a flat spinner.
           Three rings rotate on different axes inside a perspective
           box, plus a pulsing core, so depth actually changes as it
           turns rather than just scaling a 2D circle.
        ========================================================== */

        .orb-3d {

            position: absolute;
            top: 22px;
            right: 46px;
            width: 110px;
            height: 110px;
            perspective: 700px;
            z-index: 2;

        }


        .orb-3d .ring {

            position: absolute;
            inset: 0;
            border-radius: 50%;
            border: 2px solid transparent;
            transform-style: preserve-3d;

        }


        .orb-3d .ring.r1 {

            border-top-color: var(--violet);
            border-bottom-color: var(--violet);
            animation: spinX 7s linear infinite;

        }


        .orb-3d .ring.r2 {

            border-left-color: var(--gold);
            border-right-color: var(--gold);
            animation: spinY 9s linear infinite;

        }


        .orb-3d .ring.r3 {

            border-top-color: var(--cyan);
            border-left-color: var(--cyan);
            animation: spinXY 12s linear infinite;

        }


        .orb-3d .core {

            position: absolute;
            inset: 32%;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(240,220,180,0.95), rgba(130,125,255,0.35) 70%);
            box-shadow: 0 0 24px rgba(130,125,255,0.55), 0 0 40px rgba(240,185,93,0.25);
            animation: corePulse 3s ease-in-out infinite;

        }


        @keyframes spinX {
            from { transform: rotateX(0deg); }
            to { transform: rotateX(360deg); }
        }

        @keyframes spinY {
            from { transform: rotateY(0deg); }
            to { transform: rotateY(360deg); }
        }

        @keyframes spinXY {
            from { transform: rotateX(0deg) rotateY(0deg); }
            to { transform: rotateX(360deg) rotateY(360deg); }
        }

        @keyframes corePulse {
            0%, 100% { transform: scale(1); opacity: 0.9; }
            50% { transform: scale(1.15); opacity: 1; }
        }


        @media (max-width: 768px) {

            .orb-3d { display: none; }

        }


        @keyframes backgroundFloat {

            0% { transform: translate3d(0,0,0); }
            50% { transform: translate3d(-25px,20px,0); }
            100% { transform: translate3d(0,0,0); }

        }


        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }


        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif;
        }


        [data-testid="stSidebar"] {

            background: linear-gradient(180deg, rgba(20,20,34,0.98), rgba(13,14,25,0.98));
            border-right: 1px solid var(--border);

        }


        [data-testid="stSidebar"]::before {

            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 30% 20%, rgba(130,125,255,0.10), transparent 35%);
            pointer-events: none;

        }


        .hero {

            position: relative;
            padding: 2rem 2.2rem;
            border-radius: 24px;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(32,34,53,0.90), rgba(18,20,33,0.82));
            border: 1px solid var(--border);
            box-shadow: 0 25px 80px rgba(0,0,0,0.35);
            transform-style: preserve-3d;
            animation: heroFloat 8s ease-in-out infinite;

        }


        .hero::before {

            content: "";
            position: absolute;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(130,125,255,0.16), transparent 70%);
            right: -100px;
            top: -200px;
            animation: orbMove 10s ease-in-out infinite alternate;

        }


        .hero::after {

            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(240,185,93,0.10), transparent 70%);
            left: -80px;
            bottom: -140px;

        }


        @keyframes heroFloat {

            0%, 100% { transform: perspective(1200px) rotateX(0deg) rotateY(0deg) translateY(0); }
            50% { transform: perspective(1200px) rotateX(0.8deg) rotateY(-0.8deg) translateY(-4px); }

        }


        @keyframes orbMove {

            from { transform: translate3d(0,0,0); }
            to { transform: translate3d(-40px,60px,40px); }

        }


        .brand-row {

            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            gap: 18px;

        }


        .brand-icon {

            width: 58px;
            height: 58px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.9rem;
            background: linear-gradient(135deg, rgba(130,125,255,0.9), rgba(85,214,255,0.55));
            box-shadow: 0 15px 40px rgba(130,125,255,0.35), inset 0 1px 1px rgba(255,255,255,0.25);
            transform: translateZ(40px);

        }


        .brand-title {

            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(1.8rem, 3vw, 2.7rem);
            font-weight: 700;
            letter-spacing: -0.05em;
            color: var(--text);

        }


        .brand-subtitle {

            position: relative;
            z-index: 2;
            margin-top: 1rem;
            color: var(--text-muted);
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.7;

        }


        .status-row {

            position: relative;
            z-index: 2;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 1.3rem;

        }


        .status-pill {

            padding: 7px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.04);
            color: var(--text-muted);

        }


        .status-pill.ready {

            color: var(--green);
            border-color: rgba(99,230,162,0.25);
            background: rgba(99,230,162,0.06);

        }


        .status-dot {

            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-right: 6px;
            background: var(--green);
            box-shadow: 0 0 12px var(--green);
            animation: pulse 2s infinite;

        }


        @keyframes pulse {

            0%,100% { opacity: 1; }
            50% { opacity: 0.4; }

        }


        /* Sidebar step cards — real bordered containers (st.container(border=True)),
           not manually-opened/closed divs, so they actually wrap their widgets. */
        [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {

            background: rgba(255,255,255,0.025);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2px;
            transition: transform 0.25s ease, border 0.25s ease, box-shadow 0.25s ease;

        }


        [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {

            transform: perspective(900px) translateY(-3px) rotateX(1deg);
            border-color: rgba(130,125,255,0.3);
            box-shadow: 0 15px 35px rgba(0,0,0,0.25);

        }


        .step-header {

            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;

        }


        .step-number {

            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 700;
            background: var(--surface-3);
            border: 1px solid var(--border);
            color: var(--text-muted);
            flex-shrink: 0;

        }


        .step-number.done {

            background: linear-gradient(135deg, var(--gold), #d69532);
            color: #1a1207;
            border: none;
            box-shadow: 0 0 18px rgba(240,185,93,0.35);

        }


        .step-number.active {

            background: linear-gradient(135deg, var(--violet), var(--cyan));
            color: white;
            border: none;
            box-shadow: 0 0 18px rgba(130,125,255,0.4);

        }


        .step-title {

            font-weight: 700;
            font-size: 0.9rem;
            color: var(--text);

        }


        .step-description {

            font-size: 0.73rem;
            color: var(--text-muted);
            margin-top: 2px;

        }


        /* Chat cards share the same bordered-container selector as sidebar
           cards above — main-content instances get the "lift" hover instead
           of the rotate, so message cards don't tilt while you're reading. */
        [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {

            background: rgba(23,25,39,0.70);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            box-shadow: 0 15px 45px rgba(0,0,0,0.18);
            backdrop-filter: blur(15px);
            transition: transform 0.25s ease, box-shadow 0.25s ease;

        }


        [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:hover {

            transform: translateY(-2px);
            box-shadow: 0 20px 55px rgba(0,0,0,0.28);

        }


        .message-role {

            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 8px;

        }


        .message-role.user { color: var(--violet-2); }
        .message-role.assistant { color: var(--gold); }


        .source-card {

            padding: 13px;
            border-radius: 12px;
            margin-bottom: 10px;
            background: rgba(255,255,255,0.025);
            border: 1px solid var(--border);
            border-left: 3px solid var(--gold);

        }


        .source-name {

            font-weight: 700;
            color: var(--text);
            margin-bottom: 4px;

        }


        .source-score {

            font-size: 0.75rem;
            color: var(--text-muted);

        }


        .stButton button {

            border-radius: 12px;
            font-weight: 600;
            border: 1px solid rgba(130,125,255,0.35);
            background: linear-gradient(135deg, rgba(130,125,255,0.22), rgba(85,214,255,0.10));
            transition: transform 0.2s ease, box-shadow 0.2s ease;

        }


        .stButton button:hover {

            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 10px 30px rgba(130,125,255,0.2);

        }


        [data-testid="stChatInput"] {

            border-radius: 18px;
            border: 1px solid rgba(130,125,255,0.20);
            background: rgba(30,32,47,0.92);
            box-shadow: 0 15px 45px rgba(0,0,0,0.25);

        }


        /* Radio buttons default to Streamlit's red accent, which
           clashes with the violet/gold palette — retint to match. */
        [data-testid="stSidebar"] [data-baseweb="radio"] div:first-child {

            border-color: var(--violet) !important;

        }

        [data-testid="stSidebar"] [data-baseweb="radio"] div:first-child > div {

            background-color: var(--violet) !important;

        }


        @media (max-width: 768px) {

            .hero { padding: 1.4rem; }
            .brand-title { font-size: 1.7rem; }

        }

        </style>
        """
    )


inject_css()


# =============================================================================
# COMPONENTS
# =============================================================================

def render_step(
    number,
    title,
    description,
    done=False,
    active=False,
):

    state = ""

    if done:
        state = "done"

    elif active:
        state = "active"

    render_html(
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
        """
    )


def render_message(role, content):

    label = (
        "INTELLIGENCE"
        if role == "assistant"
        else "YOU"
    )

    with st.container(border=True):

        render_html(
            f"""
            <div class="message-role {role}">
                ◈ {label}
            </div>
            """
        )

        st.markdown(content)


# =============================================================================
# API
# =============================================================================

api_key = get_api_key()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown("## ◈ Vault Control")

    # -------------------------------------------------------------------------
    # VAULT
    # -------------------------------------------------------------------------

    with st.container(border=True):

        source_choice = st.radio(
            "Vault Source",
            [
                "Sample vault",
                "Upload Markdown files",
            ],
            label_visibility="collapsed",
        )

        if source_choice == "Sample vault":

            active_vault = SAMPLE_VAULT
            st.session_state.uploaded_signature = None

        else:

            uploaded_files = st.file_uploader(
                "Upload Markdown files",
                type=["md"],
                accept_multiple_files=True,
            )

            if uploaded_files:

                signature = hashlib.sha256(
                    "".join(
                        f"{file.name}-{file.size}"
                        for file in uploaded_files
                    ).encode()
                ).hexdigest()

                if signature != st.session_state.uploaded_signature:

                    saved = save_uploaded_files(
                        uploaded_files
                    )

                    st.session_state.uploaded_signature = signature

                    st.toast(
                        f"{saved} note(s) loaded",
                        icon="📚",
                    )

            active_vault = UPLOAD_VAULT

        vault_stats = get_vault_stats(
            active_vault
        )

        vault_ready = vault_stats["files"] > 0

        render_step(
            1,
            "Select Vault",
            (
                f"{vault_stats['files']} Markdown notes ready"
                if vault_ready
                else "Choose a vault to continue"
            ),
            done=vault_ready,
            active=not vault_ready,
        )

        if vault_ready:

            st.caption(
                f"📄 {vault_stats['files']} files "
                f"• {format_size(vault_stats['size'])}"
            )


    # -------------------------------------------------------------------------
    # BACKEND
    # -------------------------------------------------------------------------

    with st.container(border=True):

        backend_label = st.radio(
            "Embedding Engine",
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
            backend == "local"
            or bool(api_key)
        )

        render_step(
            2,
            "Embedding Engine",
            (
                "Gemini cloud embeddings"
                if backend == "gemini"
                else "Local private embeddings"
            ),
            done=backend_ready,
            active=not backend_ready,
        )

        if backend == "gemini" and not api_key:

            st.warning(
                "Gemini API key is not configured. "
                "Switch to Local or configure GEMINI_API_KEY in "
                "Settings → Secrets."
            )


    # -------------------------------------------------------------------------
    # INDEX
    # -------------------------------------------------------------------------

    chunk_count = get_chunk_count()

    current_fingerprint = vault_stats["fingerprint"]

    metadata = (
        st.session_state.index_metadata
    )

    index_ready = (
        metadata is not None
        and metadata.get("fingerprint")
        == current_fingerprint
        and metadata.get("backend")
        == backend
        and chunk_count > 0
    )

    with st.container(border=True):

        render_step(
            3,
            "Knowledge Index",
            (
                f"{chunk_count} chunks ready"
                if index_ready
                else "Build semantic knowledge index"
            ),
            done=index_ready,
            active=vault_ready and backend_ready and not index_ready,
        )

        if st.button(
            "⚡ Build / Rebuild Index",
            use_container_width=True,
            disabled=not (
                vault_ready
                and backend_ready
            ),
        ):

            with st.spinner(
                "Reading notes and building vector intelligence..."
            ):

                try:

                    count = build_index(
                        str(active_vault),
                        backend=backend,
                        api_key=api_key,
                    )

                    if count > 0:

                        st.session_state.index_metadata = {

                            "fingerprint":
                                get_vault_fingerprint(
                                    active_vault
                                ),

                            "backend":
                                backend,

                            "chunks":
                                count,

                        }

                        st.session_state.messages = []

                        st.success(
                            f"Index ready: {count} chunks"
                        )

                        st.rerun()

                    else:

                        st.warning(
                            "No chunks were created."
                        )

                except Exception as error:

                    st.error(
                        "Index build failed."
                    )

                    with st.expander(
                        "Technical details"
                    ):
                        st.exception(error)


    # -------------------------------------------------------------------------
    # RETRIEVAL
    # -------------------------------------------------------------------------

    st.divider()

    st.markdown(
        "### Retrieval Intelligence"
    )

    top_k = st.slider(
        "Evidence chunks",
        MIN_TOP_K,
        MAX_TOP_K,
        DEFAULT_TOP_K,
    )


    # -------------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------------

    st.divider()

    if index_ready:

        render_html(
            """
            <div class="status-pill ready">
                <span class="status-dot"></span>
                Knowledge Index Online
            </div>
            """
        )

    else:

        render_html(
            """
            <div class="status-pill">
                ○ Index Offline
            </div>
            """
        )


    # -------------------------------------------------------------------------
    # CLEAR CHAT
    # -------------------------------------------------------------------------

    if st.session_state.messages:

        st.divider()

        if st.button(
            "Clear Conversation",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.rerun()


    # -------------------------------------------------------------------------
    # USER MANUAL
    # -------------------------------------------------------------------------

    st.divider()

    with st.expander("📖 User Manual"):

        st.markdown(
            """
**1. Select Vault** — use the sample vault, or switch to
"Upload Markdown files" and drop in your own `.md` notes.

**2. Embedding Engine**
- *Gemini*: better quality, needs `GEMINI_API_KEY` set in
  `Settings → Secrets` (cloud) or `.streamlit/secrets.toml` (local).
- *Local sentence-transformers*: no key needed, runs on-device.

**3. Knowledge Index** — click **⚡ Build / Rebuild Index** any time
you change vault, switch backend, or edit your notes. The app
detects when the index is stale and tells you.

**Asking questions** — once the index is ready, type in the chat
box below, or use a quick-start button. Every answer opens an
**Evidence Used** panel showing exactly which note chunks it came
from, so you can verify or trace it back.

**Troubleshooting**
| Symptom | Fix |
|---|---|
| "Gemini API key is not configured" | Add the key in Secrets, or switch to Local |
| Build button disabled | Finish steps 1 and 2 above it first |
| "No chunks were created" | The selected folder has no readable `.md` files |
| "index is not ready" on a question | Rebuild the index — vault or backend changed |

A full deployment guide (Streamlit Community Cloud, `requirements.txt`,
`.gitignore`) ships alongside this app as `USER_MANUAL.md`.
            """
        )


# =============================================================================
# MAIN HERO
# =============================================================================

status_class = (
    "ready"
    if index_ready
    else ""
)

status_text = (
    "Knowledge index online"
    if index_ready
    else "Knowledge index needs building"
)


render_html(
    f"""
    <div class="hero">

        <div class="orb-3d">
            <div class="ring r1"></div>
            <div class="ring r2"></div>
            <div class="ring r3"></div>
            <div class="core"></div>
        </div>

        <div class="brand-row">

            <div class="brand-icon">
                ◈
            </div>

            <div>

                <div class="brand-title">
                    Obsidian Vault Intelligence
                </div>

            </div>

        </div>


        <div class="brand-subtitle">

            Turn your Markdown vault into an intelligent,
            searchable knowledge system.

            Every answer is generated from retrieved evidence
            inside your vault.

        </div>


        <div class="status-row">

            <div class="status-pill {status_class}">
                {status_text}
            </div>

            <div class="status-pill">
                {vault_stats["files"]} Notes
            </div>

            <div class="status-pill">
                {chunk_count} Knowledge Chunks
            </div>

            <div class="status-pill">
                {backend.title()} Engine
            </div>

        </div>

    </div>
    """
)


st.write("")


# =============================================================================
# CHAT HISTORY
# =============================================================================

for message in st.session_state.messages:

    render_message(
        message["role"],
        message["content"],
    )


# =============================================================================
# EMPTY STATE
# =============================================================================

if not st.session_state.messages:

    if index_ready:

        st.info(
            "Your knowledge system is ready. "
            "Ask a question or try one of the examples below."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            if st.button(
                "📚 Summarize my notes",
                use_container_width=True,
            ):

                st.session_state.quick_question = (
                    "Give me a concise summary of the most important "
                    "information in this vault."
                )

        with col2:

            if st.button(
                "🔍 Find key concepts",
                use_container_width=True,
            ):

                st.session_state.quick_question = (
                    "What are the most important concepts in these notes?"
                )

        with col3:

            if st.button(
                "🧠 Explain connections",
                use_container_width=True,
            ):

                st.session_state.quick_question = (
                    "What important connections exist between the notes?"
                )

    else:

        st.info(
            "Complete the three steps in Vault Control to activate "
            "your knowledge intelligence system."
        )


# =============================================================================
# QUESTION INPUT
# =============================================================================

quick_question = st.session_state.pop(
    "quick_question",
    None,
)

question = st.chat_input(
    "Ask something about your knowledge vault..."
)

if quick_question:
    question = quick_question


# =============================================================================
# RAG PIPELINE
# =============================================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    render_message(
        "user",
        question,
    )

    chunks = []


    # -------------------------------------------------------------------------
    # VALIDATE
    # -------------------------------------------------------------------------

    if not index_ready:

        answer = (
            "The knowledge index is not ready for the currently selected "
            "vault and embedding engine. Build the index first."
        )


    # -------------------------------------------------------------------------
    # RETRIEVE + GENERATE
    # -------------------------------------------------------------------------

    else:

        try:

            with st.spinner(
                "Searching knowledge graph..."
            ):

                chunks = retrieve(
                    question,
                    top_k=top_k,
                    api_key=api_key,
                )


            if not chunks:

                answer = (
                    "I could not find relevant evidence in the vault "
                    "to answer this reliably."
                )

            else:

                with st.spinner(
                    "Generating grounded answer..."
                ):

                    answer = generate_answer(
                        question,
                        chunks,
                        api_key=api_key,
                    )


        except Exception as error:

            answer = (
                "Something went wrong while processing your question."
            )

            chunks = []

            with st.expander(
                "Technical details"
            ):

                st.exception(error)


    # -------------------------------------------------------------------------
    # DISPLAY ANSWER
    # -------------------------------------------------------------------------

    render_message(
        "assistant",
        answer,
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


    # -------------------------------------------------------------------------
    # SOURCES
    # -------------------------------------------------------------------------

    if chunks:

        with st.expander(
            f"◈ Evidence Used — {len(chunks)} Retrieved Chunks"
        ):

            for index, chunk in enumerate(
                chunks,
                start=1,
            ):

                source = html.escape(
                    str(
                        chunk.get(
                            "source",
                            "Unknown source",
                        )
                    )
                )

                score = chunk.get(
                    "score",
                    None,
                )

                text = str(
                    chunk.get(
                        "text",
                        "",
                    )
                )

                score_text = (
                    f"Similarity distance: {float(score):.3f}"
                    if score is not None
                    else "Retrieved evidence"
                )

                render_html(
                    f"""
                    <div class="source-card">

                        <div class="source-name">

                            {index}. {source}

                        </div>

                        <div class="source-score">

                            {score_text}

                        </div>

                    </div>
                    """
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


# =============================================================================
# FOOTER
# =============================================================================

st.write("")
st.divider()

render_html(
    """
    <div style="text-align:center; color:var(--text-muted); font-size:0.75rem; padding-bottom:1rem;">
        ◈ Obsidian Vault Intelligence — answers are generated only from
        retrieved evidence in your vault.
    </div>
    """
)
