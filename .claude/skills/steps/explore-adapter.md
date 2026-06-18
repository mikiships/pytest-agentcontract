# Explore: New Adapter Work

When building or modifying an adapter (LangGraph, LlamaIndex, OpenAI Agents SDK):

1. Read the existing adapter closest to your target: `src/agentcontract/adapters/`
2. Current adapter entry points:
   - LangGraph: `record_graph(graph, recorder)` wraps `invoke()` / `ainvoke()`
   - LlamaIndex: `record_agent(agent, recorder)` wraps `chat()`, `achat()`, `query()`, and `aquery()`
   - OpenAI Agents SDK: `record_runner(recorder)` wraps `Runner.run()` and `Runner.run_sync()`
   - Public imports are lazy-loaded through `src/agentcontract/adapters/__init__.py`
3. Each adapter change should validate the target/recorder, return an unpatch function, and extract turns/tool calls into the recorder.
4. Check the target SDK's actual call pattern:
   - What function/method makes the LLM call?
   - What's the response shape?
   - Where do tool calls appear in the response?
5. Match the recording format to `src/agentcontract/types.py` Turn/ToolCall types
6. Add focused tests or an `examples/` update when adding behavior. Show record + assert/replay only where the current project workflow supports it.
