"""Framework adapters for popular agent libraries.

Each adapter wraps a framework's execution method to record trajectories
into the agentcontract format. All adapters follow the same pattern:

    unpatch = record_<thing>(target, recorder)
    # ... run your agent ...
    unpatch()

Available adapters:
    - langgraph: LangGraph CompiledGraph (invoke/ainvoke)
    - llamaindex: LlamaIndex AgentRunner (chat/query)
    - openai_agents: OpenAI Agents SDK Runner (run/run_sync)
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy-load adapters to avoid importing heavy framework dependencies."""
    _lazy = {
        "record_graph": "agentcontract.adapters.langgraph",
        "record_agent": "agentcontract.adapters.llamaindex",
        "record_runner": "agentcontract.adapters.openai_agents",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["record_graph", "record_agent", "record_runner"]
