# Explore: New Adapter Work

When building or modifying an adapter (LangGraph, LlamaIndex, OpenAI Agents SDK):

1. Read the existing adapter closest to your target: `src/agentcontract/adapters/`
2. Each adapter needs:
   - A recorder wrapper that intercepts the SDK call path and writes `AgentRun` turns
   - Registration in `src/agentcontract/adapters/__init__.py`
   - Focused unit tests in `tests/unit/test_adapters.py`
3. Replay is centralized in `src/agentcontract/replay/engine.py` and exposed through pytest fixtures. Do not add replay hooks inside individual adapters unless the task explicitly asks for one.
4. Check the target SDK's actual call pattern:
   - What function/method makes the LLM call?
   - What's the response shape?
   - Where do tool calls appear in the response?
5. Match the recording format to `src/agentcontract/types.py` `Turn` and `ToolCall` types
6. Update README/examples only when the public workflow changes. Examples should show recording through the adapter and replay/assertion through the shared pytest fixtures or replay engine.
