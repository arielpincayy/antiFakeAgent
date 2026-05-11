import streamlit as st
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FactCheck AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0b0c0f !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2130 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Typography ── */
h1, h2, h3, .syne { font-family: 'Syne', sans-serif !important; }

/* ── Hero header ── */
.hero {
    padding: 2.5rem 0 1.5rem;
    display: flex;
    align-items: baseline;
    gap: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.03em;
    color: #e8e4dc;
    line-height: 1;
}
.hero-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4ade80;
    border: 1px solid #4ade8044;
    background: #4ade8010;
    padding: 3px 10px;
    border-radius: 2px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Source pills ── */
.source-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.6rem;
}
.pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 4px 12px;
    border-radius: 2px;
    text-transform: uppercase;
    border: 1px solid;
}
.pill-arxiv    { color: #f97316; border-color: #f9731644; background: #f9731610; }
.pill-scholar  { color: #60a5fa; border-color: #60a5fa44; background: #60a5fa10; }
.pill-openalex { color: #a78bfa; border-color: #a78bfa44; background: #a78bfa10; }
.pill-web      { color: #34d399; border-color: #34d39944; background: #34d39910; }

/* ── Input area ── */
.stTextArea textarea {
    background: #141720 !important;
    border: 1px solid #252a3a !important;
    border-radius: 4px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 1rem !important;
    resize: vertical !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px #4ade8020 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #4ade80 !important;
    color: #0b0c0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.65rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #86efac !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #4ade8030 !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Secondary button ── */
[data-testid="baseButton-secondary"] > button,
.secondary-btn > button {
    background: transparent !important;
    color: #e8e4dc !important;
    border: 1px solid #252a3a !important;
}
.secondary-btn > button:hover {
    background: #141720 !important;
    border-color: #4ade8066 !important;
}

/* ── Result panel ── */
.result-card {
    background: #111318;
    border: 1px solid #1e2130;
    border-left: 3px solid #4ade80;
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
}
.result-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #4ade80;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 0 0 0.8rem;
}

/* ── Verdict banner ── */
.verdict-true {
    background: #052e16;
    border: 1px solid #166534;
    border-left: 4px solid #4ade80;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
}
.verdict-false {
    background: #2d0a0a;
    border: 1px solid #7f1d1d;
    border-left: 4px solid #f87171;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
}
.verdict-mixed {
    background: #1c1407;
    border: 1px solid #78350f;
    border-left: 4px solid #fbbf24;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
}

/* ── Chat ── */
.chat-user {
    background: #141720;
    border: 1px solid #252a3a;
    border-radius: 4px 4px 4px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    position: relative;
}
.chat-user::before {
    content: 'YOU';
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: #60a5fa;
    display: block;
    margin-bottom: 0.3rem;
}
.chat-assistant {
    background: #0f1c14;
    border: 1px solid #1a3026;
    border-radius: 4px 4px 0 4px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.chat-assistant::before {
    content: 'FACTCHECK AI';
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: #4ade80;
    display: block;
    margin-bottom: 0.3rem;
}

/* ── Metrics ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.metric-box {
    flex: 1;
    background: #141720;
    border: 1px solid #1e2130;
    border-radius: 3px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-box .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #4ade80;
    line-height: 1;
}
.metric-box .lbl {
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    color: #6b7280;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Sidebar items ── */
.sidebar-claim {
    background: #0f1218;
    border: 1px solid #1e2130;
    border-radius: 3px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.75rem;
    color: #9ca3af;
    cursor: pointer;
    transition: border-color 0.15s;
}
.sidebar-claim:hover { border-color: #4ade8066; color: #e8e4dc; }

/* ── Divider ── */
hr { border-color: #1e2130 !important; margin: 1.5rem 0 !important; }

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: #4ade80 !important; }

/* ── Analysis sources list ── */
.src-item {
    border-top: 1px solid #1e2130;
    padding: 0.6rem 0;
    font-size: 0.78rem;
    color: #9ca3af;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.src-item a { color: #60a5fa; text-decoration: none; }
.src-badge {
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0b0c0f; }
::-webkit-scrollbar-thumb { background: #252a3a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "rag": None,
    "chat_history": [],
    "current_claim": "",
    "final_report": "",
    "analysis": [],
    "elapsed": 0.0,
    "history_log": [],   # list of {claim, report, elapsed}
    "agent": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Lazy agent init ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def get_agent():
    from app.agent import ResearchAgent
    return ResearchAgent()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="syne" style="font-size:1.1rem;font-weight:700;color:#e8e4dc;margin-bottom:0.2rem">FactCheck AI</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.68rem;color:#6b7280;letter-spacing:0.1em;text-transform:uppercase;margin-top:0">Research · Verify · Report</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p style="font-size:0.65rem;letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-bottom:0.6rem">Sources Active</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="source-row" style="margin:0 0 1rem">
        <span class="pill pill-arxiv">arXiv</span>
        <span class="pill pill-scholar">Scholar</span>
        <span class="pill pill-openalex">OpenAlex</span>
        <span class="pill pill-web">Web</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history_log:
        st.markdown('<p style="font-size:0.65rem;letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-bottom:0.6rem">Previous Checks</p>', unsafe_allow_html=True)
        for i, entry in enumerate(reversed(st.session_state.history_log[-6:])):
            short = entry["claim"][:60] + ("…" if len(entry["claim"]) > 60 else "")
            st.markdown(f'<div class="sidebar-claim">#{len(st.session_state.history_log)-i} {short}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("＋ New Claim", key="new_btn"):
        st.session_state.rag = None
        st.session_state.chat_history = []
        st.session_state.current_claim = ""
        st.session_state.final_report = ""
        st.session_state.analysis = []
        st.session_state.elapsed = 0.0
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-title">FactCheck AI</span>
    <span class="hero-badge">v2.0 · RAG Enabled</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="source-row">
    <span class="pill pill-arxiv">arXiv</span>
    <span class="pill pill-scholar">Google Scholar</span>
    <span class="pill pill-openalex">OpenAlex</span>
    <span class="pill pill-web">Web Search</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 – Claim input
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.rag is None:

    claim = st.text_area(
        "Enter a claim or news headline to verify",
        placeholder="e.g.  'A new AI model can predict earthquakes 72 hours in advance with 95% accuracy.'",
        height=110,
        label_visibility="visible",
    )

    col_run, col_ex = st.columns([2, 3])

    with col_run:
        run_btn = st.button("🔍  Verify Claim", key="run")

    # Example claims
    examples = [
        "mRNA vaccines alter human DNA",
        "Coffee consumption reduces the risk of type 2 diabetes",
        "5G towers are causing COVID-19 infections",
    ]
    with col_ex:
        selected = st.selectbox("Or try an example →", [""] + examples, label_visibility="collapsed")

    if selected and not run_btn:
        claim = selected

    if run_btn and claim.strip():
        agent = get_agent()
        from app.rag import RAGStore

        with st.spinner("Researching across scientific databases and the web…"):
            t0 = time.time()
            report, analysis = agent.run(claim.strip())
            elapsed = time.time() - t0

        rag = RAGStore(agent.agent)
        rag.build(analysis, claim.strip(), report)

        st.session_state.rag = rag
        st.session_state.current_claim = claim.strip()
        st.session_state.final_report = report
        st.session_state.analysis = analysis
        st.session_state.elapsed = elapsed
        st.session_state.chat_history = []
        st.session_state.history_log.append({
            "claim": claim.strip(),
            "report": report,
            "elapsed": elapsed,
        })
        st.rerun()

    elif run_btn:
        st.warning("Please enter a claim to verify.")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 – Results + RAG chat
# ═══════════════════════════════════════════════════════════════════════════════
else:
    claim     = st.session_state.current_claim
    report    = st.session_state.final_report
    analysis  = st.session_state.analysis
    elapsed   = st.session_state.elapsed

    # ── Claim recap ──
    st.markdown(f"""
    <div style="background:#141720;border:1px solid #252a3a;border-radius:4px;padding:0.9rem 1.2rem;margin-bottom:1.2rem">
        <span style="font-size:0.6rem;letter-spacing:0.15em;color:#6b7280;text-transform:uppercase">Claim verified</span><br>
        <span style="font-size:0.95rem;color:#e8e4dc">{claim}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ──
    n_sources = len(analysis)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box"><div class="val">{n_sources}</div><div class="lbl">Sources</div></div>
        <div class="metric-box"><div class="val">{elapsed:.1f}s</div><div class="lbl">Research time</div></div>
        <div class="metric-box"><div class="val">RAG</div><div class="lbl">Chat ready</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab_report, tab_sources, tab_chat = st.tabs(["📋 Report", "📚 Sources", "💬 Ask AI"])

    # ── Tab: Report ──────────────────────────────────────────────────────────
    with tab_report:
        st.markdown('<div class="result-card"><h4>Full Analysis Report</h4>', unsafe_allow_html=True)
        st.markdown(report)
        st.markdown('</div>', unsafe_allow_html=True)

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                "⬇ Download Report",
                data=f"# FactCheck Report\n\n**Claim:** {claim}\n\n---\n\n{report}",
                file_name="factcheck_report.md",
                mime="text/markdown",
            )

    # ── Tab: Sources ─────────────────────────────────────────────────────────
    with tab_sources:
        if analysis:
            source_colors = {
                "arXiv": "#f97316", "Google Scholar": "#60a5fa",
                "OpenAlex": "#a78bfa", "Web": "#34d399",
            }
            for item in analysis:
                src   = item.get("source", "Unknown")
                color = source_colors.get(src, "#9ca3af")
                title = item.get("title", "No title")
                summ  = item.get("summary", "")[:220] + ("…" if len(item.get("summary","")) > 220 else "")
                link  = item.get("link", "#")
                cred  = item.get("credibility", "—")

                st.markdown(f"""
                <div style="border:1px solid #1e2130;border-left:3px solid {color};
                            border-radius:3px;padding:0.9rem 1.1rem;margin-bottom:0.7rem;background:#111318">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem">
                        <div style="flex:1">
                            <span style="font-size:0.58rem;letter-spacing:0.12em;
                                         text-transform:uppercase;color:{color}">{src}</span><br>
                            <span style="font-size:0.85rem;color:#e8e4dc;font-family:'Syne',sans-serif;
                                         font-weight:600">{title}</span>
                            <p style="font-size:0.78rem;color:#9ca3af;margin:0.4rem 0 0.2rem">{summ}</p>
                            <a href="{link}" target="_blank"
                               style="font-size:0.7rem;color:#60a5fa;text-decoration:none">
                               ↗ View source</a>
                        </div>
                        <div style="text-align:right;white-space:nowrap">
                            <span style="font-size:0.6rem;letter-spacing:0.1em;color:#6b7280;
                                         text-transform:uppercase">credibility</span><br>
                            <span style="font-size:0.9rem;font-family:'Syne',sans-serif;
                                         font-weight:700;color:{color}">{cred}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No sources were collected.")

    # ── Tab: Chat ─────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("""
        <div style="background:#0f1c14;border:1px solid #1a3026;border-radius:4px;
                    padding:0.7rem 1rem;margin-bottom:1.2rem;font-size:0.78rem;color:#6ee7b7">
            ⚡ RAG active — I can answer specific questions about this analysis using the retrieved sources.
        </div>
        """, unsafe_allow_html=True)

        # Render chat history
        for msg in st.session_state.chat_history:
            cls = "chat-user" if msg["role"] == "user" else "chat-assistant"
            st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        user_input = st.text_input(
            "Ask a question about the analysis",
            placeholder="Which sources support this claim? What does the scientific evidence say?",
            label_visibility="collapsed",
            key="chat_input",
        )

        col_ask, col_clr = st.columns([3, 1])
        with col_ask:
            ask_btn = st.button("Send →", key="ask_btn")
        with col_clr:
            if st.button("Clear chat", key="clr_btn"):
                st.session_state.chat_history = []
                st.rerun()

        if ask_btn and user_input.strip():
            with st.spinner("Searching context…"):
                response = st.session_state.rag.chat(
                    st.session_state.chat_history,
                    user_input.strip(),
                    claim,
                )
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()