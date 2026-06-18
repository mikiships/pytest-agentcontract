# Explore: New Adapter Work

When building or modifying an adapter (LangGraph, LlamaIndex, OpenAI Agents SDK):

1. Read the existing adapter closest to your target: `src/agentcontract/adapters/`
2. Current adapter entrypoints are:
   - `record_graph(graph, recorder)` for LangGraph `invoke` / `ainvoke`
   - `record_agent(agent, recorder)` for LlamaIndex `chat` / `achat` / `query` / `aquery`
   - `record_runner(recorder)` for OpenAI Agents SDK `Runner.run` / `Runner.run_sync`
3. Each adapter wraps framework execution methods, records turns/tool calls into the recorder, and returns an unpatch function that restores original methods.
4. Public adapter exports lazy-load through `src/agentcontract/adapters/__init__.py`; update the lazy map and `__all__` for new public adapter functions.
5. Check the target SDK's actual call pattern:
   - What function/method makes the LLM call?
   - What's the response shape?
   - Where do tool calls appear in the response?
6. Match the recording format to `src/agentcontract/types.py` Turn/ToolCall types.
7. Add focused tests for record/unpatch behavior. Add an `examples/` update only when the task asks for user-facing adapter examples.
