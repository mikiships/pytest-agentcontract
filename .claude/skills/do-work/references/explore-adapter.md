# Explore: Framework Adapter Work

Use this reference when building or modifying an adapter for LangGraph, LlamaIndex, OpenAI Agents SDK, or another agent framework.

1. Read the closest existing adapter in `src/agentcontract/adapters/`.
2. Existing adapters wrap framework execution methods and record trajectories through `agentcontract.recorder.core.Recorder`.
3. Each adapter should:
   - Validate that the provided target exposes the methods being wrapped.
   - Return an unpatch function that restores original methods.
   - Record turns, tool calls, arguments, results, and latency when available.
   - Register the public entrypoint in `src/agentcontract/adapters/__init__.py`.
4. Check the target SDK's actual call pattern:
   - Which function or method runs the agent?
   - What response shape is returned?
   - Where do tool calls and tool outputs appear?
5. Match recorded data to the structures in `src/agentcontract/types.py`.
6. Add focused tests in `tests/unit/test_adapters.py`.
7. If adding a new public example, place it under `examples/` and keep it minimal.
