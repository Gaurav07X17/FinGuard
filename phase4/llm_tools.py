"""
llm_tools.py

Function schemas for OpenAI-style function-calling. These are lightweight
representations of the calculation functions exposed to an LLM.

Each tool is represented as a dict suitable for passing to the OpenAI
function-calling API (name, description, parameters JSON Schema).
"""

CALCULATE_SHARPE = {
    "name": "calculate_sharpe",
    "description": "Calculate the Sharpe ratio for a series of periodic returns.",
    "parameters": {
        "type": "object",
        "properties": {
            "returns": {"type": "array", "items": {"type": "number"}},
            "risk_free_rate": {"type": "number", "description": "Annualized risk-free rate (optional)"},
            "periods_per_year": {"type": "integer", "default": 252}
        },
        "required": ["returns"]
    }
}

CALCULATE_VOLATILITY = {
    "name": "calculate_volatility",
    "description": "Calculate annualized volatility from periodic returns.",
    "parameters": {
        "type": "object",
        "properties": {
            "returns": {"type": "array", "items": {"type": "number"}},
            "periods_per_year": {"type": "integer", "default": 252}
        },
        "required": ["returns"]
    }
}

CALCULATE_VAR = {
    "name": "calculate_var",
    "description": "Calculate historical or parametric Value-at-Risk (VaR) for a returns series.",
    "parameters": {
        "type": "object",
        "properties": {
            "returns": {"type": "array", "items": {"type": "number"}},
            "confidence_level": {"type": "number", "default": 0.95},
            "method": {"type": "string", "enum": ["historical", "parametric"], "default": "historical"}
        },
        "required": ["returns"]
    }
}

ALL_TOOLS = [CALCULATE_SHARPE, CALCULATE_VOLATILITY, CALCULATE_VAR]
