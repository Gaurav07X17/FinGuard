"""
streamlit_app.py

The single-page Streamlit app that ties Phases 2-4 together:
    1. Upload a portfolio file (CSV/XLSX)
    2. Confirm the auto-guessed column mapping
    3. See an instant, auto-computed risk dashboard (no LLM call — pure math)
    4. Ask plain-English risk questions, answered via LLM function-calling

Design choices:
- All state lives in st.session_state — no database, no persistence
  across sessions, matching the project's non-goals (no multi-user
  accounts, no complex backend).
- The dashboard in step 3 calls risk_metrics.py functions directly,
  bypassing the LLM entirely. This is deliberate: the four metrics are
  deterministic math, so computing them via an LLM tool call would be
  slower, cost a rate-limit request, and add no value. The LLM is
  reserved for what it's actually good at — answering open-ended
  questions about what the numbers mean.
- Session-based rate limiting caps how many questions one browser
  session can ask. This protects the free-tier Gemini key from being
  exhausted by a single runaway session (accidental loop, or — since
  this app may eventually be deployed publicly — someone else using
  the link). It's intentionally simple: an in-memory counter, no
  accounts, consistent with the project's non-goals.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import streamlit as st

from ingest import load_portfolio_file, list_excel_sheets, IngestionError
from mapping import guess_column_mapping, validate_mapping
from convert import dataframe_to_returns, ConversionError
from risk_metrics import (
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_var,
    calculate_max_drawdown,
)
from nlp_engine import ask_question

MAX_QUESTIONS_PER_SESSION = 15

st.set_page_config(page_title="FinGuard — Portfolio Risk Analysis", page_icon="🛡️", layout="centered")

# ---------------------------------------------------------------------------
# Styling — injected once, referenced via CSS custom properties throughout.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #F1E9D8;
    --surface: #EAE0C8;
    --surface-2: #E2D5B6;
    --border: #C9B98F;
    --text: #2A2118;
    --text-dim: #6B5D45;
    --accent: #1F3D2B;
    --accent-soft: rgba(31, 61, 43, 0.09);
    --brass: #A6832E;
    --brass-soft: rgba(166, 131, 46, 0.14);
    --risk-low: #4B7A52;
    --risk-mod: #A6782E;
    --risk-high: #A34430;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--text) !important; }

/* Streamlit auto-adds a link/anchor icon to every heading (native or raw HTML)
   for deep-linking. It reads as a stray cursor/icon artifact in a polished UI,
   so it's hidden globally here rather than left to appear on hover. */
[data-testid="stHeaderActionElements"] { display: none !important; }

.fg-hero {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1.6rem 0 0.4rem 0;
    border-bottom: 2px solid var(--accent);
    margin-bottom: 1.6rem;
}
.fg-hero .badge {
    width: 52px; height: 52px;
    border-radius: 50%;
    background: var(--brass-soft);
    border: 1.5px solid var(--brass);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 14px rgba(166, 131, 46, 0.2), inset 0 1px 0 rgba(255,255,255,0.3);
    flex-shrink: 0;
}
.fg-hero .fg-title {
    font-family: 'Fraunces', serif;
    font-size: 2.1rem; margin: 0; line-height: 1.1; font-weight: 600;
    color: var(--text);
}
.fg-hero p { color: var(--text-dim); margin: 0.2rem 0 0 0; font-size: 0.98rem; }

.fg-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(60, 45, 20, 0.12), inset 0 1px 0 rgba(255,255,255,0.4);
}

.fg-metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.9rem;
    margin: 0.4rem 0 1.2rem 0;
}
.fg-metric-card {
    background: linear-gradient(180deg, var(--surface), var(--surface-2));
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1rem 1.1rem;
    box-shadow: 0 4px 14px rgba(60, 45, 20, 0.14), inset 0 1px 0 rgba(255,255,255,0.4);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.fg-metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(60, 45, 20, 0.2);
}
.fg-metric-label {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-dim); margin-bottom: 0.35rem;
}
.fg-metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem; font-weight: 600; color: var(--text);
    line-height: 1;
}
.fg-metric-tag {
    display: inline-block; margin-top: 0.5rem;
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
    padding: 0.15rem 0.55rem; border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
}
.tag-low  { background: rgba(75, 122, 82, 0.14); color: var(--risk-low); }
.tag-mod  { background: rgba(166, 120, 46, 0.14); color: var(--risk-mod); }
.tag-high { background: rgba(163, 68, 48, 0.14); color: var(--risk-high); }

.fg-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--brass); margin-bottom: 0.3rem;
}

.stChatMessage { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border);
    border-radius: 8px;
    padding: 0.5rem;
    background: var(--surface);
}

.stButton > button {
    background: var(--accent) !important;
    color: #F1E9D8 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 14px rgba(31, 61, 43, 0.25);
}
.stButton > button:hover { box-shadow: 0 6px 20px rgba(31, 61, 43, 0.4); }

.fg-quota { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--text-dim); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
for key, default in [
    ("df", None),
    ("mapping", None),
    ("returns", None),
    ("filename", None),
    ("chat_history", []),
    ("questions_asked", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_portfolio():
    """Clear everything downstream of file upload — used when a new file is loaded."""
    st.session_state.df = None
    st.session_state.mapping = None
    st.session_state.returns = None
    st.session_state.filename = None
    st.session_state.chat_history = []
    st.session_state.questions_asked = 0


def risk_tag(level: str) -> str:
    """Map a qualitative risk level to a small CSS tag class + label."""
    mapping = {
        "low": ("tag-low", "Lower risk"),
        "moderate": ("tag-mod", "Moderate risk"),
        "high": ("tag-high", "Higher risk"),
    }
    cls, label = mapping.get(level, ("tag-mod", "—"))
    return f'<span class="fg-metric-tag {cls}">{label}</span>'


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="fg-hero">
    <div class="badge">🛡️</div>
    <div>
        <div class="fg-title">FinGuard</div>
        <p>Upload a portfolio, ask a risk question in plain English, get a calculated answer.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Step 1: File upload
# ---------------------------------------------------------------------------
st.markdown('<div class="fg-section-label">Step 1</div>', unsafe_allow_html=True)
st.subheader("Upload your portfolio")

uploaded_file = st.file_uploader(
    "CSV or XLSX file with at least a date column and a price/value column",
    type=["csv", "xlsx"],
    label_visibility="collapsed",
)

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    reset_portfolio()
    st.session_state.filename = uploaded_file.name

    try:
        sheet_name = 0
        if uploaded_file.name.lower().endswith(".xlsx"):
            uploaded_file.seek(0)
            sheets = list_excel_sheets(uploaded_file, uploaded_file.name)
            if len(sheets) > 1:
                sheet_name = st.selectbox("Multiple sheets found — pick one:", sheets)
            uploaded_file.seek(0)

        df = load_portfolio_file(uploaded_file, uploaded_file.name, sheet_name=sheet_name)
        st.session_state.df = df
        st.success(f"Loaded {len(df):,} rows from {uploaded_file.name}.")
    except IngestionError as e:
        st.error(str(e))

# ---------------------------------------------------------------------------
# Step 2: Column mapping confirmation
# ---------------------------------------------------------------------------
if st.session_state.df is not None and st.session_state.returns is None:
    st.markdown('<div class="fg-section-label">Step 2</div>', unsafe_allow_html=True)
    st.subheader("Confirm column mapping")

    df = st.session_state.df
    with st.container():
        st.dataframe(df.head(5), use_container_width=True)

    guess = guess_column_mapping(df)
    columns = list(df.columns)

    st.caption("FinGuard guessed the following mapping. Adjust anything that looks wrong.")

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox(
            "Date column *",
            options=columns,
            index=columns.index(guess["mapping"]["date"]) if guess["mapping"]["date"] else 0,
        )
        price_col = st.selectbox(
            "Price / Value column *",
            options=columns,
            index=columns.index(guess["mapping"]["price"]) if guess["mapping"]["price"] else 0,
        )
    with col2:
        symbol_options = ["(none — single portfolio value)"] + columns
        symbol_default = guess["mapping"]["symbol"] or symbol_options[0]
        symbol_col = st.selectbox("Symbol column (multi-asset only)", options=symbol_options,
                                   index=symbol_options.index(symbol_default) if symbol_default in symbol_options else 0)

        quantity_options = ["(none)"] + columns
        quantity_default = guess["mapping"]["quantity"] or quantity_options[0]
        quantity_col = st.selectbox("Quantity column (multi-asset only)", options=quantity_options,
                                     index=quantity_options.index(quantity_default) if quantity_default in quantity_options else 0)

    st.caption("* Required. Only set Symbol/Quantity if your file has one row per date "
               "per holding (multi-asset). Leave both as 'none' for a single portfolio value column.")

    if st.button("Confirm mapping and calculate returns", type="primary"):
        confirmed_mapping = {
            "date": date_col,
            "price": price_col,
            "symbol": None if symbol_col == symbol_options[0] else symbol_col,
            "quantity": None if quantity_col == "(none)" else quantity_col,
        }
        try:
            validate_mapping(df, confirmed_mapping)
            returns = dataframe_to_returns(df, confirmed_mapping)
            st.session_state.mapping = confirmed_mapping
            st.session_state.returns = returns
            st.rerun()
        except (ValueError, ConversionError) as e:
            st.error(str(e))

# ---------------------------------------------------------------------------
# Step 3: Auto-computed risk dashboard (no LLM — pure calculation)
# ---------------------------------------------------------------------------
if st.session_state.returns is not None:
    returns = st.session_state.returns

    st.markdown('<div class="fg-section-label">Step 3</div>', unsafe_allow_html=True)
    st.subheader("Risk dashboard")
    st.caption(f"Calculated instantly from {len(returns)} periods of returns data — no AI call needed for this part.")

    vol = calculate_volatility(returns)
    sharpe = calculate_sharpe_ratio(returns)
    var = calculate_var(returns)
    mdd = calculate_max_drawdown(returns)

    vol_level = "low" if vol["value"] < 0.10 else "moderate" if vol["value"] < 0.20 else "high"
    sharpe_level = "high" if sharpe["value"] < 0 else "moderate" if sharpe["value"] < 1 else "low"
    var_level = "low" if var["value"] < 0.02 else "moderate" if var["value"] < 0.04 else "high"
    mdd_level = "low" if mdd["value"] < 0.10 else "moderate" if mdd["value"] < 0.20 else "high"

    st.markdown(f"""
    <div class="fg-metric-grid">
        <div class="fg-metric-card">
            <div class="fg-metric-label">Volatility (annualized)</div>
            <div class="fg-metric-value">{vol['value']:.1%}</div>
            {risk_tag(vol_level)}
        </div>
        <div class="fg-metric-card">
            <div class="fg-metric-label">Sharpe Ratio</div>
            <div class="fg-metric-value">{sharpe['value']:.2f}</div>
            {risk_tag(sharpe_level)}
        </div>
        <div class="fg-metric-card">
            <div class="fg-metric-label">Value at Risk (95%, 1-day)</div>
            <div class="fg-metric-value">{var['value']:.1%}</div>
            {risk_tag(var_level)}
        </div>
        <div class="fg-metric-card">
            <div class="fg-metric-label">Max Drawdown</div>
            <div class="fg-metric-value">{mdd['value']:.1%}</div>
            {risk_tag(mdd_level)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Portfolio value over time"):
        cumulative_value = (1 + returns).cumprod()
        st.line_chart(cumulative_value, use_container_width=True)

    if st.button("Upload a different portfolio"):
        reset_portfolio()
        st.rerun()

    # -----------------------------------------------------------------------
    # Step 4: Chat interface
    # -----------------------------------------------------------------------
    st.markdown('<div class="fg-section-label">Step 4</div>', unsafe_allow_html=True)
    st.subheader("Ask a risk question")

    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.questions_asked
    st.markdown(
        f'<span class="fg-quota">{remaining} of {MAX_QUESTIONS_PER_SESSION} questions remaining this session</span>',
        unsafe_allow_html=True,
    )

    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        st.warning(
            "No Gemini API key found. Add GEMINI_API_KEY to your Streamlit secrets "
            "(.streamlit/secrets.toml locally, or the Secrets manager on Streamlit Community Cloud)."
        )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if remaining <= 0:
        st.info(
            f"You've reached the {MAX_QUESTIONS_PER_SESSION}-question limit for this session. "
            f"This protects the free-tier API key from being exhausted. Refresh the page to start a new session."
        )
    else:
        question = st.chat_input("e.g. How risky is this portfolio? What's my worst-case daily loss?")

        if question and api_key:
            st.session_state.questions_asked += 1
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Calculating..."):
                    try:
                        result = ask_question(question, returns, api_key=api_key)
                        answer = result["answer"]
                        st.write(answer)
                        if result["tool_calls"]:
                            with st.expander("How this was calculated"):
                                for call in result["tool_calls"]:
                                    st.write(f"**{call['name']}**({call['input']})")
                                    st.json(call["output"])
                    except (ValueError, RuntimeError) as e:
                        answer = f"Something went wrong: {e}"
                        st.error(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        elif question and not api_key:
            st.error("Add an API key before asking questions — see the warning above.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "FinGuard calculates volatility, Sharpe ratio, Value at Risk, and max drawdown from "
    "your uploaded data. It does not provide investment advice."
)
