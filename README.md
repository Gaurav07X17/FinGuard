# FinGuard

Portfolio managers and self-directed investors often know something feels "off" about a
portfolio's risk profile long before they can quantify it. FinGuard closes that gap: upload a
portfolio, ask a plain-English question — *"how risky is this?"*, *"what's my worst-case
daily loss?"*, *"is my Sharpe ratio good?"* — and get back a calculated, explained answer,
not a dashboard you have to interpret yourself.

## What it does

- Accepts a portfolio as a CSV or Excel upload — single portfolio value over time, or
  multi-asset holdings (symbol, price, quantity)
- Auto-detects which columns are date/price/symbol/quantity, and shows you the guessed
  mapping to confirm before anything is calculated — no silent misreads of your data
- Answers risk questions in natural language via LLM function-calling, backed by real
  calculations — not approximations or LLM guesswork
- Explains *why* a number matters, not just what it is
- Shows exactly which calculation ran and what it returned, for anyone who wants to verify
  the answer rather than take it on faith

## Metrics calculated

| Metric | What it tells you |
|---|---|
| Volatility | How much the portfolio's returns swing, annualized |
| Sharpe Ratio | Return earned per unit of risk taken, vs. a risk-free baseline |
| Value at Risk (VaR) | The most you'd expect to lose in a single day, at a given confidence level |
| Max Drawdown | The worst peak-to-trough decline the portfolio has experienced |

## How it works

1. **Upload** — CSV or XLSX, single-series or multi-asset
2. **Confirm mapping** — FinGuard guesses which columns are date/price/symbol/quantity;
   you review and adjust before anything downstream runs
3. **Ask** — plain-English questions get routed to the right calculation via LLM
   function-calling; the model never states a number without calling a tool first
4. **Verify** — every answer can be expanded to show the exact tool called, its inputs,
   and its raw output

## Status

All five build phases are implemented and tested:

- ✅ Calculation engine — four risk metrics, tested against synthetic and edge-case data
- ✅ File ingestion — CSV/XLSX loading, validation, size/row limits
- ✅ Column mapping — fuzzy auto-detection with user confirmation, never auto-applied silently
- ✅ NLP layer — LLM function-calling (Google Gemini, free tier) with guardrails against
  hallucinated figures and prompt injection
- ✅ Streamlit UI — full upload → mapping → chat flow

**Remaining before this is "done":** a security review pass, testing against a wider range
of real-world file formats, and deployment to Streamlit Community Cloud.

## Project structure

```
FinGuard/
├── src/
│   ├── risk_metrics.py     # Core calculation engine (4 metrics)
│   ├── ingest.py           # File loading + validation
│   ├── mapping.py          # Column auto-detection
│   ├── convert.py          # Prices/holdings -> returns series
│   ├── llm_tools.py        # Tool schemas for LLM function-calling
│   ├── tool_wrappers.py    # Safe dispatch layer between LLM and calculations
│   └── nlp_engine.py       # Orchestrates question -> tool call -> answer
├── streamlit_app.py        # The app itself
├── tests/
│   ├── test_risk_metrics.py
│   └── test_tool_wrappers.py
├── docs/
│   └── nlp_manual_testing.md
├── .streamlit/
│   └── secrets.toml.example
├── requirements.txt
└── README.md
```

## Scope

This is a deliberately right-sized build — a single Streamlit app, not a multi-service
platform. Explicitly out of scope: web scraping, vector databases, RAG, multi-user accounts,
and multi-service infrastructure (no Docker, no Kubernetes). The goal is a focused tool that
does one thing — portfolio risk Q&A — well.

## Tech stack

Python, Pandas/NumPy/SciPy for calculations, Google Gemini (free tier) for the
natural-language layer via function-calling, Streamlit for the interface.

## Running locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml with a free Gemini API key from https://aistudio.google.com/apikey
streamlit run streamlit_app.py
```
