# Prompts and usage guidance for LLM integration

This file documents recommended prompt patterns and guardrails when using an LLM
with the function-calling tools defined in phase4/llm_tools.py and the
wrappers in phase4/tool_wrappers.py.

Guidelines
- Use OpenAI-style function-calling with the `ALL_TOOLS` schema. Provide the
  `returns` array as raw numeric values (floats) representing periodic returns.
- Do not allow the model to free-text numeric results for final answers. Instead,
  prefer that the model select the appropriate `function` and return a JSON
  payload. The application should then render the `interpretation` for
  human-readable output.
- Validate inputs server-side: always run the values through the wrappers
  in phase4/tool_wrappers.py which perform type coercion and sanity checks.

Prompt template (example)
- System prompt: include guardrails (see system_prompt.txt)
- User: "Given these daily returns [0.01, -0.005, ...], calculate the 95% VaR"
- Assistant (model): should call the `calculate_var` function with the
  `returns` parameter set to the list of numbers and `confidence_level`=0.95.

Security & anti-hallucination
- Never accept or display numeric conclusions that are not corroborated by the
  function output. If the model provides free-text numbers, ignore them and
  prefer the function response.
- Sanitize and clamp inputs: reject extremely large arrays (>100k) or non-numeric
  entries before calling the function.

Testing
- Unit tests should mock the LLM function-calling behavior (i.e., directly call
  the wrapper functions with representative inputs) so CI does not require
  external API keys.
