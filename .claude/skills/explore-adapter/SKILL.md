---
name: explore-adapter
description: Guides pytest-agentcontract adapter work for LangGraph, LlamaIndex, OpenAI Agents SDK, and new framework integrations. Use when modifying src/agentcontract/adapters, adapter exports, recorder integration, framework call interception, Turn and ToolCall mapping, examples, or adapter tests.
---

# Explore: Adapter Work

When building or modifying an adapter (LangGraph, LlamaIndex, OpenAI Agents SDK):

1. Read the existing adapter closest to your target in `src/agentcontract/adapters/`.
2. Current adapters wrap framework execution to record trajectories:
   - `langgraph.py`: `record_graph` wraps `invoke` and `ainvoke`
   - `llamaindex.py`: `record_agent` wraps `chat`, `achat`, `query`, and `aquery`
   - `openai_agents.py`: `record_runner` wraps `Runner.run` and `Runner.run_sync`
3. Register public adapter helpers in `src/agentcontract/adapters/__init__.py` and keep lazy imports intact.
4. Check the target SDK's actual call pattern:
   - What function/method makes the LLM call?
   - What's the response shape?
   - Where do tool calls appear in the response?
5. Match the recording format to `src/agentcontract/types.py` `Turn` and `ToolCall` types.
6. Cover adapter behavior in `tests/unit/test_adapters.py` with mock framework objects, not live SDK calls.
7. If adding a new framework adapter, add a minimal example under `examples/` showing record, replay, and assertion flow.
