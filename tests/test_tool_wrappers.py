"""
Tests for tool_wrappers.py and llm_tools.py — the parts of Phase 4 that don't
require a live API call. nlp_engine.py (the actual LLM orchestration) needs
a real ANTHROPIC_API_KEY and is intended to be verified manually / in Phase 7
integration testing rather than here, to avoid unit tests that cost API calls
or fail in CI without secrets configured.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from llm_tools import get_tool_definitions, get_tool_names
from tool_wrappers import execute_tool, execute_tool_calls

np.random.seed(7)
SAMPLE_RETURNS = pd.Series(np.random.normal(0.0004, 0.011, 250))


def test_tool_definitions_shape():
    tools = get_tool_definitions()
    assert len(tools) == 4
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


def test_tool_names_match_wrappers():
    expected = {
        "calculate_volatility",
        "calculate_sharpe_ratio",
        "calculate_var",
        "calculate_max_drawdown",
    }
    assert get_tool_names() == expected


def test_execute_tool_valid_no_args():
    result = execute_tool("calculate_volatility", SAMPLE_RETURNS, {})
    assert "value" in result
    assert "interpretation" in result
    assert "error" not in result


def test_execute_tool_valid_with_args():
    result = execute_tool("calculate_sharpe_ratio", SAMPLE_RETURNS, {"risk_free_rate": 0.03})
    assert "value" in result
    assert isinstance(result["value"], float)


def test_execute_tool_none_input_defaults_to_empty():
    result = execute_tool("calculate_max_drawdown", SAMPLE_RETURNS, None)
    assert "value" in result


def test_execute_tool_unknown_tool_name():
    result = execute_tool("delete_all_data", SAMPLE_RETURNS, {})
    assert "error" in result
    assert "Unknown tool" in result["error"]


def test_execute_tool_invalid_argument_value():
    result = execute_tool("calculate_var", SAMPLE_RETURNS, {"confidence": 10})
    assert "error" in result


def test_execute_tool_hallucinated_parameter():
    result = execute_tool("calculate_volatility", SAMPLE_RETURNS, {"not_a_real_param": 1})
    assert "error" in result
    assert "Invalid arguments" in result["error"]


def test_execute_tool_calls_batch():
    calls = [
        {"id": "a", "name": "calculate_volatility", "input": {}},
        {"id": "b", "name": "calculate_var", "input": {"confidence": 0.9}},
    ]
    results = execute_tool_calls(calls, SAMPLE_RETURNS)
    assert len(results) == 2
    assert results[0]["tool_use_id"] == "a"
    assert results[1]["tool_use_id"] == "b"
    assert "value" in results[0]["output"]
    assert "value" in results[1]["output"]


def test_execute_tool_calls_batch_with_one_failure():
    calls = [
        {"id": "a", "name": "calculate_volatility", "input": {}},
        {"id": "b", "name": "not_a_real_tool", "input": {}},
    ]
    results = execute_tool_calls(calls, SAMPLE_RETURNS)
    assert "value" in results[0]["output"]
    assert "error" in results[1]["output"]


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in test_fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nAll {len(test_fns)} tests passed.")
