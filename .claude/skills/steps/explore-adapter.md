# Explore: New Adapter Work

When building or modifying an adapter (LangGraph, LlamaIndex, OpenAI Agents SDK):

1. Read the existing adapter closest to your target: `src/agentcontract/adapters/`
2. Each adapter currently needs:
   - A recording wrapper that intercepts the framework execution method(s)
   - Extraction of messages, tool calls, and tool results into recorder turns
   - An `unpatch` function that restores original SDK methods
   - Lazy export registration in `src/agentcontract/adapters/__init__.py`
3. Check the target SDK's actual call pattern:
   - What function/method makes the LLM call?
   - What's the response shape?
   - Where do tool calls appear in the response?
4. Match the recording format to `src/agentcontract/types.py` Turn/ToolCall types
5. Add or update focused adapter tests under `tests/unit/`
6. If the adapter is user-facing, write a minimal example in `examples/` showing record + replay + assert
