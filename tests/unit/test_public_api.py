"""Tests for package-level public exports."""

import importlib
from types import SimpleNamespace

import pytest

import agentcontract
from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AgentContractConfig
from agentcontract.recorder.core import Recorder
from agentcontract.replay.engine import ReplayEngine


def test_package_lazy_getattr_imports_expected_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    imported: list[str] = []

    def fake_import_module(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(Recorder=sentinel)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert agentcontract.__getattr__("Recorder") is sentinel
    assert imported == ["agentcontract.recorder.core"]


def test_package_lazy_getattr_raises_for_unknown_name() -> None:
    with pytest.raises(AttributeError, match="no attribute 'UnknownSymbol'"):
        agentcontract.__getattr__("UnknownSymbol")


def test_package_all_exports_are_stable() -> None:
    reloaded = importlib.reload(agentcontract)

    assert reloaded.__all__ == [
        "Recorder",
        "ReplayEngine",
        "AssertionEngine",
        "AgentContractConfig",
    ]


def test_subpackage_exports_reference_public_classes() -> None:
    assertions = importlib.reload(importlib.import_module("agentcontract.assertions"))
    recorder = importlib.reload(importlib.import_module("agentcontract.recorder"))
    replay = importlib.reload(importlib.import_module("agentcontract.replay"))

    assert assertions.__all__ == ["AssertionEngine"]
    assert assertions.AssertionEngine is AssertionEngine
    assert recorder.__all__ == ["Recorder"]
    assert recorder.Recorder is Recorder
    assert replay.__all__ == ["ReplayEngine"]
    assert replay.ReplayEngine is ReplayEngine


def test_root_package_lazy_exports_resolve_real_types() -> None:
    assert agentcontract.Recorder is importlib.import_module("agentcontract.recorder.core").Recorder
    assert agentcontract.ReplayEngine is importlib.import_module("agentcontract.replay.engine").ReplayEngine
    assert (
        agentcontract.AssertionEngine
        is importlib.import_module("agentcontract.assertions.engine").AssertionEngine
    )
    assert (
        agentcontract.AgentContractConfig
        is importlib.import_module("agentcontract.config").AgentContractConfig
    )
