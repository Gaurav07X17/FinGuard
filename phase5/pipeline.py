"""
Phase 5 pipeline: orchestrate ingestion -> convert -> compute metrics.

Usage (CLI):
    python -m phase5.pipeline --file path/to/data.csv --metrics sharpe,var,vol --var_conf 0.95

This pipeline:
- Loads a file (CSV/XLSX bytes) using phase3.ingest.ingest_file
- Autoguesses mapping with phase3.mapping.auto_guess_mapping
- Converts price column to returns via phase3.convert.df_price_column_to_returns
- Computes requested metrics using deterministic wrappers in phase4.tool_wrappers
"""
from typing import List, Dict, Any, Optional
import json
import argparse
import os

from phase3.ingest import ingest_file, NaNError, MissingColumns
from phase3.mapping import auto_guess_mapping
from phase3.convert import df_price_column_to_returns
from phase4.tool_wrappers import calculate_sharpe, calculate_volatility, calculate_var


def compute_metrics_from_returns(
    returns_list: List[float],
    metrics: List[str],
    var_confidence: float = 0.95,
    var_method: str = "historical",
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """Compute requested metrics from a returns list."""
    results: Dict[str, Any] = {}
    if "sharpe" in metrics:
        results["sharpe"] = calculate_sharpe(
            returns_list, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )
    if "vol" in metrics or "volatility" in metrics:
        results["volatility"] = calculate_volatility(
            returns_list, periods_per_year=periods_per_year
        )
    if "var" in metrics:
        results["var"] = calculate_var(
            returns_list, confidence_level=var_confidence, method=var_method
        )
    return results


def run_pipeline_on_file(
    filepath: str,
    metrics: List[str],
    nan_policy: str = "reject",
    required_columns: Optional[List[str]] = None,
    var_confidence: float = 0.95,
    var_method: str = "historical",
) -> Dict[str, Any]:
    """Run pipeline for a single file path. Returns a JSON-serializable dict."""
    if required_columns is None:
        required_columns = []
    # read bytes
    with open(filepath, "rb") as f:
        b = f.read()
    # ingestion
    try:
        ingested = ingest_file(b, filename=os.path.basename(filepath), required_columns=required_columns, nan_policy=nan_policy)
    except MissingColumns as e:
        return {"error": "missing_columns", "message": str(e)}
    except NaNError as e:
        return {"error": "nan_error", "message": str(e)}
    except Exception as e:
        return {"error": "ingest_failed", "message": str(e)}
    df = ingested["df"]
    # guess mapping and convert to returns
    mapping = auto_guess_mapping(list(df.columns))
    try:
        returns_series = df_price_column_to_returns(df, mapping)
    except Exception as e:
        return {"error": "convert_failed", "message": str(e)}
    # make returns list
    returns_list = list(returns_series.astype(float).tolist())
    # compute metrics
    try:
        results = compute_metrics_from_returns(
            returns_list,
            metrics,
            var_confidence=var_confidence,
            var_method=var_method,
        )
    except Exception as e:
        return {"error": "compute_failed", "message": str(e)}
    return {"file": os.path.basename(filepath), "mapping": mapping, "results": results}


def cli():
    parser = argparse.ArgumentParser(description="Phase5: pipeline runner")
    parser.add_argument("--file", required=True, help="Path to CSV/XLSX file")
    parser.add_argument("--metrics", default="sharpe,var,vol", help="Comma-separated metrics: sharpe,var,vol")
    parser.add_argument("--nan_policy", default="reject", choices=["reject", "drop", "impute"])
    parser.add_argument("--var_conf", type=float, default=0.95, help="VaR confidence level")
    parser.add_argument("--var_method", default="historical", choices=["historical", "parametric"])
    args = parser.parse_args()

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    out = run_pipeline_on_file(
        args.file,
        metrics=metrics,
        nan_policy=args.nan_policy,
        var_confidence=args.var_conf,
        var_method=args.var_method,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cli()
