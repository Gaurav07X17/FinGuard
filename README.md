Here is a production-grade **`README.md`** tailored specifically for **FinGuard**. It covers the architecture, setup, CLI/UI usage, testing, and deployment steps clearly.

---

```markdown name=README.md
# 🛡️ FinGuard — AI-Driven Financial Risk Engine

**FinGuard** is a production-grade, modular financial risk engine that bridges raw financial time-series data, rigorous statistical risk metrics, and Large Language Model (LLM) query interpretation. 

By enforcing strict separation between **data ingestion**, **deterministic mathematical calculations**, and **natural language tooling**, FinGuard guarantees mathematical accuracy while offering an intuitive conversational interface.

---

## 🏗️ Architecture Overview

FinGuard is built with a layered, decoupled design contract:

```text
  [ User Upload / Prompt ]
             │
             ▼
┌──────────────────────────┐
│  Phase 5: Streamlit UI   │ ──► Interactive Web Dashboard & Chat Interface
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Phase 3: Data Ingestion │ ──► CSV/XLSX Parser, Auto-Mapping, & Return Conversion
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Phase 4: LLM Tooling    │ ──► OpenAI-Style Function Schemas & Validated Wrappers
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Phase 2: Risk Engine    │ ──► Pure, Validated Risk Math (Sharpe, Vol, VaR, Drawdown)
└──────────────────────────┘

```

---

## ⚡ Key Features

* **Deterministic Math Engine:** Pure NumPy/SciPy statistical models for Volatility, Sharpe Ratio, Value at Risk (Historical & Parametric), and Max Drawdown.
* **Safe LLM Function Calling:** LLM tools return standardized payloads `{"value": float, "interpretation": str}` to prevent mathematical hallucinations.
* **Robust Data Pipeline:** Auto-detects delimiters, character encodings, and column structures for `.csv` and `.xlsx` files with strict `NaN` rejection policies.
* **Interactive Dashboard:** Built with Streamlit for file previewing, parameter tweaking, and interactive visual reporting.
* **CI/CD Enforced:** Matrix unit testing across Python 3.10 and 3.11 with zero live LLM API keys required for testing.

---

## 🚀 Quick Start

### 1. Prerequisites

* **Python 3.10** or higher
* `pip` and `venv`

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone repository
git clone [https://github.com/your-username/finguard.git](https://github.com/your-username/finguard.git)
cd finguard

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

```

---

## 💻 Running the Application

### Launch Streamlit Web UI (Phase 5)

To launch the interactive dashboard:

```bash
streamlit run phase5/streamlit_app.py

```

Open [http://localhost:8501](http://localhost:8501) in your browser to upload files, map columns, run calculations, and export JSON reports.

---

## 🧪 Testing

FinGuard maintains strict testing standards. Run the entire test suite locally:

```bash
# Run all unit tests quietly
pytest -q

# Run specific phase test suites
pytest phase3/tests/test_ingest.py
pytest phase4/tests/test_llm_tools.py

```

---

## 📊 Core Risk Metrics

| Metric | Code Module | Description |
| --- | --- | --- |
| **Annualized Sharpe Ratio** | `phase4.tool_wrappers.calculate_sharpe` | Risk-adjusted return using configurable risk-free rates and periods. |
| **Annualized Volatility** | `phase4.tool_wrappers.calculate_volatility` | Standard deviation of periodic returns scaled by periods per year. |
| **Value at Risk (VaR)** | `phase4.tool_wrappers.calculate_var` | 1-day portfolio risk calculation via **Historical Percentile** or **Parametric (Gaussian)** methods. |
| **Maximum Drawdown** | `phase2.risk_metrics.max_drawdown` | Peak-to-trough decline percentage over the given timeframe. |

---

## 📂 Repository Structure

```text
finguard/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI matrix workflow
├── phase2/                    # Statistical calculation engine
│   └── risk_metrics.py
├── phase3/                    # Ingestion, auto-mapping, & return conversion
│   ├── ingest.py
│   ├── mapping.py
│   ├── convert.py
│   └── tests/
├── phase4/                    # LLM tools, prompts, & function wrappers
│   ├── llm_tools.py
│   ├── tool_wrappers.py
│   ├── system_prompt.txt
│   └── tests/
├── phase5/                    # User interface & orchestrator
│   └── streamlit_app.py
├── requirements-dev.txt       # Project dependencies
└── README.md

```

---

## 🛡️ Guardrails & Limits

* **Maximum File Size:** 10 MB limit on raw price file uploads.
* **Row Cap:** 100,000 max row ingestion limit.
* **Strict NaN Handling:** Default `'reject'` policy halts processing on incomplete series to protect against metric corruption.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```
