# Explore: New Adapter Work

When building or modifying an adapter (LangGraph, LlamaIndex, OpenAI Agents SDK):

1. Read the existing adapter closest to your target: `src/agentcontract/adapters/`
2. Current adapter entry points are `record_graph`, `record_agent`, and `record_runner`
3. Each adapter should:
   - Wrap the target framework execution method(s)
   - Record turns/tool calls into the shared recorder format
   - Return an unpatch function that restores original methods
   - Lazy-load through `src/agentcontract/adapters/__init__.py` when exported publicly
4. Check the target SDK's actual call pattern:
   - What function/method makes the LLM call?
   - What's the response shape?
   - Where do tool calls appear in the response?
5. Match the recording format to `src/agentcontract/types.py` Turn/ToolCall types
6. Do not add adapter-specific replay behavior unless the task explicitly defines that contract
7. Write or update tests under `tests/unit/`; add examples only when the task asks for user-facing sample code
