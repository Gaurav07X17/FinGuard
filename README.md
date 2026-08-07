# FinGuard

Portfolio managers and self-directed investors often know something feels "off" about a
portfolio's risk profile long before they can quantify it. FinGuard closes that gap: upload a
portfolio, ask a plain-English question — *"how risky is this?"*, *"what's my worst-case
daily loss?"*, *"how would this have performed in a bad month?"* — and get back a calculated,
explained answer, not a dashboard you have to interpret yourself.

## What it does

- Accepts a portfolio as an Excel upload (holdings, prices, or returns)
- Answers risk questions in natural language, backed by real calculations — not
  approximations or LLM guesswork
- Explains *why* a number matters, not just what it is

## Metrics calculated

| Metric | What it tells you |
|---|---|
| Volatility | How much the portfolio's returns swing, annualized |
| Sharpe Ratio | Return earned per unit of risk taken, vs. a risk-free baseline |
| Value at Risk (VaR) | The most you'd expect to lose in a single day, at a given confidence level |
| Max Drawdown | The worst peak-to-trough decline the portfolio has experienced |

## Status

Early build. Calculation engine (the core risk math) is complete and tested. File ingestion,
the natural-language layer, and the UI are in progress.

## Project structure

```
FinGuard/
├── src/
│   └── risk_metrics.py       # Core calculation engine
├── tests/
│   └── test_risk_metrics.py  # Validation tests
├── requirements.txt
└── README.md
```

## Scope

This is a deliberately right-sized build — a single Streamlit app, not a multi-service
platform. Explicitly out of scope: web scraping, vector databases, RAG, and multi-user
accounts. The goal is a focused tool that does one thing — portfolio risk Q&A — well.

## Tech stack

Python, Pandas/NumPy for calculations, LLM function-calling for the natural-language layer,
Streamlit for the interface.
