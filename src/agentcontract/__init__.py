"""pytest-agentcontract: Deterministic CI tests for LLM agent trajectories."""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies and speed up pytest plugin loading.
# The plugin.py entry point imports specific modules directly.


def __getattr__(name: str):  # noqa: ANN001
    """Lazy-load public API classes on first access."""
    _lazy = {
        "Recorder": "agentcontract.recorder.core",
        "ReplayEngine": "agentcontract.replay.engine",
        "AssertionEngine": "agentcontract.assertions.engine",
        "AgentContractConfig": "agentcontract.config",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Recorder", "ReplayEngine", "AssertionEngine", "AgentContractConfig"]
