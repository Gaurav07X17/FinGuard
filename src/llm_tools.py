"""
llm_tools.py

Defines the tool schemas the LLM uses for function-calling, in Anthropic's
tool-use format (Claude Messages API). Each tool maps 1:1 to a function in
risk_metrics.py — the LLM picks a tool + arguments based on the user's
plain-English question, and tool_wrappers.py executes it.

Design choice: schemas are static and explicit, not auto-generated from
function signatures. This is more code to maintain, but it means the
descriptions can be written for LLM comprehension (natural language,
examples of what the metric answers) rather than for Python developers.
That distinction matters a lot for tool-selection accuracy.

If swapping to a different provider's function-calling format (e.g. OpenAI),
convert TOOL_DEFINITIONS' input_schema into that provider's parameters
schema — the structure is JSON Schema either way, so the conversion is
mostly renaming keys (input_schema -> parameters).
"""

TOOL_DEFINITIONS = [
    {
        "name": "calculate_volatility",
        "description": (
            "Calculate portfolio volatility (annualized standard deviation of returns). "
            "Use this when the user asks about risk, how much the portfolio fluctuates, "
            "how volatile/stable it is, or wants a general risk measure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "annualize": {
                    "type": "boolean",
                    "description": "Whether to annualize the volatility. Default true unless the user specifically asks for daily volatility.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "calculate_sharpe_ratio",
        "description": (
            "Calculate the Sharpe ratio (risk-adjusted return per unit of volatility). "
            "Use this when the user asks about risk-adjusted performance, whether returns "
            "are 'worth' the risk taken, or explicitly mentions Sharpe ratio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "risk_free_rate": {
                    "type": "number",
                    "description": "Annualized risk-free rate as a decimal (e.g. 0.04 for 4%). Only set this if the user specifies a rate; otherwise omit to use the default.",
                },
                "annualize": {
                    "type": "boolean",
                    "description": "Whether to annualize return and volatility before computing the ratio. Default true.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "calculate_var",
        "description": (
            "Calculate Value at Risk (VaR) — the maximum expected loss over one day at a "
            "given confidence level. Use this when the user asks about worst-case loss, "
            "downside risk, 'how much could I lose', or explicitly mentions VaR."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "number",
                    "description": "Confidence level as a decimal (e.g. 0.95 for 95%, 0.99 for 99%). Only set this if the user specifies a level; otherwise omit to use the default (0.95).",
                },
                "method": {
                    "type": "string",
                    "enum": ["historical", "parametric"],
                    "description": "Estimation method. Default 'historical' unless the user asks for a normal-distribution/parametric estimate.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "calculate_max_drawdown",
        "description": (
            "Calculate maximum drawdown — the largest peak-to-trough decline in portfolio "
            "value over the observed period. Use this when the user asks about the worst "
            "historical decline, biggest loss period, or drawdown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def get_tool_definitions() -> list:
    """
    Return the list of tool schemas for use in an LLM function-calling request.

    Returns:
        list[dict]: tool definitions in Anthropic tool-use format, ready to
            pass as the `tools` parameter in a Messages API call.
    """
    return TOOL_DEFINITIONS


def get_tool_names() -> set:
    """
    Return the set of valid tool names, used by tool_wrappers.py to validate
    that an LLM-selected tool name is one we actually support before executing it.

    Returns:
        set[str]: valid tool names.
    """
    return {tool["name"] for tool in TOOL_DEFINITIONS}


def get_tool_definitions_gemini() -> list:
    """
    Return the tool schemas converted to Gemini's function-declaration format.

    Gemini's Interactions API expects each tool as a flat dict with a
    "type": "function" key and "parameters" instead of Anthropic's
    "input_schema" — otherwise the same JSON Schema content. This converts
    TOOL_DEFINITIONS once so both providers can be supported from the same
    source of truth without duplicating the schema descriptions.

    Returns:
        list[dict]: tool definitions in Gemini function-declaration format,
            ready to pass as the `tools` parameter in an Interactions API call.
    """
    gemini_tools = []
    for tool in TOOL_DEFINITIONS:
        gemini_tools.append({
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        })
    return gemini_tools
