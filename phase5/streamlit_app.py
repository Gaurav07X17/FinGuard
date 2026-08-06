"""
Simple Streamlit frontend to upload a price file, confirm mapping, choose metrics,
and run the Phase5 pipeline (ingest -> convert -> compute via deterministic wrappers).

Usage:
  streamlit run phase5/streamlit_app.py
"""
import io
import json
import pandas as pd
import streamlit as st

# import lower-level functions to avoid using CLI-only pipeline file
from phase3.ingest import ingest_file, NaNError, MissingColumns
from phase3.mapping import auto_guess_mapping
from phase3.convert import df_price_column_to_returns
from phase4.tool_wrappers import calculate_sharpe, calculate_volatility, calculate_var

st.set_page_config(page_title="FinGuard — Risk Explorer", layout="wide")

st.title("FinGuard — Risk Explorer (Streamlit)")

uploaded = st.file_uploader("Upload a CSV or XLSX price file", type=["csv", "xlsx"])
st.markdown("Defaults: Max rows 100k, Max file size 10MB. NaN policy default = reject.")

col1, col2 = st.columns([2, 1])

with col1:
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        st.write(f"Filename: {uploaded.name} — {len(file_bytes)} bytes")
        # Try ingest with minimal required columns if user provided
        try:
            ingested = ingest_file(file_bytes, filename=uploaded.name, required_columns=[], nan_policy="reject")
            df = ingested["df"]
            st.subheader("Preview")
            st.dataframe(df.head(10))
            # Auto-detect mapping
            mapping = auto_guess_mapping(list(df.columns))
            st.write("Auto-detected mapping (you can override):")
            st.write(mapping)
            # Allow user override for price column
            price_col = st.selectbox("Select price column", options=list(df.columns), index=list(df.columns).index(mapping.get("price", list(df.columns)[0])) if mapping.get("price") in df.columns else 0)
            # Optional date column selector if present
            date_col = None
            date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if date_candidates:
                date_col = st.selectbox("Select date column (optional)", options=[None] + date_candidates, index=0)
            # NaN policy
            nan_policy = st.selectbox("NaN policy", options=["reject", "drop", "impute"], index=0)
            # Metrics selector
            metrics = st.multiselect("Metrics to compute", options=["sharpe", "volatility", "var"], default=["sharpe", "var", "volatility"])
            periods = st.number_input("Periods per year (for annualization)", value=252, min_value=1)
            var_conf = st.slider("VaR confidence level", min_value=0.9, max_value=0.999, value=0.95, step=0.01)
            var_method = st.selectbox("VaR method", options=["historical", "parametric"])
            run = st.button("Run")
        except MissingColumns as e:
            st.error(f"Missing required columns: {e}")
            df = None
        except NaNError as e:
            st.error(f"NaN policy: {e}")
            df = None
        except Exception as e:
            st.error(f"Ingest failed: {e}")
            df = None

        if uploaded is not None and df is not None and st.session_state.get("run_triggered", False) is False:
            # store mapping override in session state so results persist
            st.session_state["last_df"] = df
            st.session_state["last_price_col"] = price_col
            st.session_state["last_date_col"] = date_col
            st.session_state["last_nan_policy"] = nan_policy

with col2:
    st.markdown("### Quick actions")
    if uploaded is None:
        st.info("Upload a file to enable actions.")
    else:
        if st.button("Accept auto-mapping and run"):
            st.session_state["run_triggered"] = True
            st.session_state["use_price_col"] = st.session_state.get("last_price_col")
            st.session_state["use_nan_policy"] = st.session_state.get("last_nan_policy")

# Run when triggered
if st.session_state.get("run_triggered", False):
    df = st.session_state.get("last_df")
    price_col = st.session_state.get("use_price_col", None) or st.session_state.get("last_price_col")
    nan_policy = st.session_state.get("use_nan_policy", "reject")
    if df is None or price_col not in df.columns:
        st.error("Invalid dataframe or price column.")
    else:
        # Ensure date sorting if date column provided
        date_col = st.session_state.get("last_date_col")
        if date_col and date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.sort_values(by=date_col)
            except Exception:
                st.warning("Failed to parse/sort date column; proceeding without reordering.")

        # Convert to returns
        try:
            returns = df[price_col].astype(float).pct_change().dropna().tolist()
        except Exception as e:
            st.error(f"Failed to convert prices to returns: {e}")
            returns = []

        if len(returns) == 0:
            st.error("No returns to compute. Check price column and NaN policy.")
        else:
            results = {}
            if "sharpe" in metrics:
                try:
                    results["sharpe"] = calculate_sharpe(returns, risk_free_rate=0.0, periods_per_year=int(periods))
                except Exception as e:
                    results["sharpe_error"] = str(e)
            if "volatility" in metrics or "vol" in metrics:
                try:
                    results["volatility"] = calculate_volatility(returns, periods_per_year=int(periods))
                except Exception as e:
                    results["volatility_error"] = str(e)
            if "var" in metrics:
                try:
                    results["var"] = calculate_var(returns, confidence_level=float(var_conf), method=var_method)
                except Exception as e:
                    results["var_error"] = str(e)

            st.success("Computation finished.")
            st.subheader("Results")
            st.json(results)

            # Plot returns and cumulative returns
            st.subheader("Returns plot")
            r_ser = pd.Series(returns)
            st.line_chart(r_ser)
            st.subheader("Cumulative returns")
            cum = (1 + r_ser).cumprod() - 1
            st.line_chart(cum)

            # Allow download of JSON
            result_json = json.dumps({"mapping": {"price": price_col, "date": date_col}, "results": results}, indent=2)
            st.download_button("Download results (JSON)", data=result_json, file_name=f"{uploaded.name}-results.json", mime="application/json")

            # Reset trigger
            st.session_state["run_triggered"] = False
