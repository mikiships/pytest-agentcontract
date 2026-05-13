# Explore: New Adapter Work

When building or modifying an adapter (LangGraph, LlamaIndex, OpenAI Agents SDK):

1. Read the existing adapter closest to your target: `src/agentcontract/adapters/`
2. Each adapter needs:
   - A recorder integration that wraps SDK or framework calls and records AgentRun turns
   - Registration in `src/agentcontract/adapters/__init__.py`
   - Focused coverage in `tests/unit/test_adapters.py`
3. Check the target SDK's actual call pattern:
   - What function/method makes the LLM call?
   - What's the response shape?
   - Where do tool calls appear in the response?
4. Match the recording format to `src/agentcontract/types.py` Turn/ToolCall types
5. Update README examples only when public adapter usage changes
6. Write a minimal example in `examples/` if the adapter introduces a new user-facing workflow
