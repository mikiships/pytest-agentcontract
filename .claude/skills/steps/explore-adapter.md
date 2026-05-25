# Explore: New Adapter Work

When building or modifying an adapter (LangGraph, LlamaIndex, OpenAI Agents SDK):

1. Read the existing adapter closest to your target: `src/agentcontract/adapters/`
2. Each adapter needs:
   - A recorder integration that intercepts the framework's sync and async execution methods
   - An unpatch function that restores the original methods
   - Registration in `src/agentcontract/adapters/__init__.py`
3. Check the target SDK's actual call pattern:
   - What function/method makes the LLM call?
   - What's the response shape?
   - Where do tool calls appear in the response?
4. Match the recording format to `src/agentcontract/types.py` Turn/ToolCall types
5. Add focused coverage in `tests/unit/test_adapters.py`; add or update an
   `examples/customer_support/` example only when the workflow needs user-facing
   documentation
