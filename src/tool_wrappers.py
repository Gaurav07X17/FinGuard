"""
tool_wrappers.py

Safely executes a tool call selected by the LLM. This is the trust boundary
between "text an LLM generated" and "code that actually runs" — every input
here is treated as untrusted, even though it's typically just numbers.

Design choices:
- Tool name is validated against llm_tools.get_tool_names() before dispatch.
  An LLM can't invoke arbitrary functions, only the four defined tools.
- Arguments are passed as explicit kwargs to known functions, never eval'd
  or dynamically constructed — there's no code execution surface here.
- Errors from risk_metrics.py (e.g. bad confidence value) are caught and
  returned as a structured error dict rather than raised, so the LLM can
  read the error and explain it to the user instead of the app crashing.
"""

import pandas as pd

from llm_tools import get_tool_names
from risk_metrics import (
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_var,
    calculate_max_drawdown,
)

_TOOL_FUNCTIONS = {
    "calculate_volatility": calculate_volatility,
    "calculate_sharpe_ratio": calculate_sharpe_ratio,
    "calculate_var": calculate_var,
    "calculate_max_drawdown": calculate_max_drawdown,
}


def execute_tool(tool_name: str, returns: pd.Series, tool_input: dict = None) -> dict:
    """
    Execute a single tool call against the portfolio's returns series.

    Args:
        tool_name: Name of the tool to run, as selected by the LLM. Must be
            one of the names defined in llm_tools.TOOL_DEFINITIONS.
        returns: The portfolio's returns Series (from convert.dataframe_to_returns()).
        tool_input: Arguments the LLM provided for the tool call, e.g.
            {"risk_free_rate": 0.03}. Defaults to {} if the LLM provided no
            arguments (valid — every tool's args are optional with defaults).

    Returns:
        dict: On success, the tool's normal {"value": ..., "interpretation": ...}
            output. On failure, {"error": str} describing what went wrong —
            this is returned, not raised, so the calling LLM turn can see it
            and respond appropriately rather than crashing the app.
    """
    tool_input = tool_input or {}

    if tool_name not in get_tool_names():
        return {"error": f"Unknown tool '{tool_name}'. This is not a recognized calculation."}

    func = _TOOL_FUNCTIONS[tool_name]

    try:
        result = func(returns, **tool_input)
    except TypeError as e:
        return {"error": f"Invalid arguments for '{tool_name}': {e}"}
    except ValueError as e:
        return {"error": f"Could not calculate '{tool_name}': {e}"}
    except Exception as e:
        return {"error": f"Unexpected error running '{tool_name}': {e}"}

    return result


def execute_tool_calls(tool_calls: list, returns: pd.Series) -> list:
    """
    Execute a batch of tool calls (as an LLM response may request more than one
    in a single turn, e.g. "what's my volatility and Sharpe ratio?").

    Args:
        tool_calls: list of dicts, each with at minimum {"id": str, "name": str,
            "input": dict} — matching the shape of Anthropic tool_use content
            blocks.
        returns: The portfolio's returns Series.

    Returns:
        list[dict]: one result per input tool call, each shaped as
            {"tool_use_id": str, "output": dict}, where "output" is either
            the tool's normal result or an {"error": str} dict. This shape
            is ready to be wrapped into Anthropic tool_result content blocks.
    """
    results = []
    for call in tool_calls:
        output = execute_tool(call["name"], returns, call.get("input", {}))
        results.append({"tool_use_id": call["id"], "output": output})
    return results
