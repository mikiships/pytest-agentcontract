"""Tests for pytest plugin fixtures and hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentcontract import plugin
from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AgentContractConfig, AssertionSpec, ScenarioOverride
from agentcontract.recorder.core import Recorder
from agentcontract.types import AgentRun


@dataclass
class FakeMarker:
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeNode:
    def __init__(self, name: str, markers: dict[str, FakeMarker] | None = None) -> None:
        self.name = name
        self._markers = markers or {}

    def get_closest_marker(self, name: str) -> FakeMarker | None:
        return self._markers.get(name)


class FakeConfig:
    def __init__(self, **options: Any) -> None:
        self._options = options
        self.lines: list[tuple[str, str]] = []

    def getoption(self, name: str) -> Any:
        return self._options.get(name)

    def addinivalue_line(self, key: str, value: str) -> None:
        self.lines.append((key, value))


@dataclass
class FakeRequest:
    config: FakeConfig
    node: FakeNode


class FakeOptionGroup:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def addoption(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class FakeParser:
    def __init__(self) -> None:
        self.group_name: tuple[str, str] | None = None
        self.group = FakeOptionGroup()

    def getgroup(self, name: str, description: str) -> FakeOptionGroup:
        self.group_name = (name, description)
        return self.group


def _finish_fixture(fixture_gen: Any) -> None:
    with pytest.raises(StopIteration):
        next(fixture_gen)


def _write_cassette(path: Path, scenario: str) -> Path:
    recorder = Recorder(scenario=scenario)
    with recorder.recording():
        recorder.add_turn(role="assistant", content="recorded")
    recorder.save(path)
    return path


def test_pytest_addoption_registers_expected_flags() -> None:
    parser = FakeParser()

    plugin.pytest_addoption(parser)

    assert parser.group_name == ("agentcontract", "Agent trajectory testing")
    option_names = [args[0] for args, _ in parser.group.calls]
    assert option_names == ["--ac-record", "--ac-replay", "--ac-config", "--ac-scenarios"]


def test_pytest_configure_registers_markers() -> None:
    config = FakeConfig()

    plugin.pytest_configure(config)

    assert len(config.lines) == 2
    assert config.lines[0][0] == "markers"
    assert "agentcontract(scenario)" in config.lines[0][1]
    assert "agent_scenario(name" in config.lines[1][1]


def test_ac_config_uses_explicit_path(monkeypatch, tmp_path: Path) -> None:
    expected = AgentContractConfig(version="from-file")

    def fake_from_file(cls: type[AgentContractConfig], path: Path) -> AgentContractConfig:
        assert path == tmp_path / "agentcontract.yml"
        return expected

    monkeypatch.setattr(plugin.AgentContractConfig, "from_file", classmethod(fake_from_file))
    request = FakeRequest(
        config=FakeConfig(**{"--ac-config": str(tmp_path / "agentcontract.yml")}),
        node=FakeNode("test_case"),
    )

    config = plugin.ac_config.__wrapped__(request)

    assert config is expected


def test_ac_config_discovers_defaults(monkeypatch) -> None:
    expected = AgentContractConfig(version="discovered")
    monkeypatch.setattr(
        plugin.AgentContractConfig,
        "discover",
        classmethod(lambda cls: expected),
    )
    request = FakeRequest(config=FakeConfig(**{"--ac-config": None}), node=FakeNode("test_case"))

    config = plugin.ac_config.__wrapped__(request)

    assert config is expected


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"--ac-record": True, "--ac-replay": False}, "record"),
        ({"--ac-record": False, "--ac-replay": True}, "replay"),
        ({"--ac-record": False, "--ac-replay": False}, "live"),
    ],
)
def test_ac_mode_selects_record_replay_or_live(options: dict[str, bool], expected: str) -> None:
    request = FakeRequest(config=FakeConfig(**options), node=FakeNode("test_case"))

    assert plugin.ac_mode.__wrapped__(request) == expected


def test_ac_recorder_saves_recorded_cassette_with_marker_name(tmp_path: Path) -> None:
    request = FakeRequest(
        config=FakeConfig(**{"--ac-record": True, "--ac-scenarios": str(tmp_path)}),
        node=FakeNode("ignored_name", markers={"agentcontract": FakeMarker(args=("refund-flow",))}),
    )

    fixture_gen = plugin.ac_recorder.__wrapped__(request)
    recorder = next(fixture_gen)
    recorder.add_turn(role="assistant", content="Refund approved")
    _finish_fixture(fixture_gen)

    saved = tmp_path / "refund-flow.agentrun.json"
    assert saved.exists()
    assert recorder.run.metadata.scenario == "refund-flow"


def test_ac_recorder_uses_alias_marker_name_without_saving_in_live_mode(tmp_path: Path) -> None:
    request = FakeRequest(
        config=FakeConfig(**{"--ac-record": False, "--ac-scenarios": str(tmp_path)}),
        node=FakeNode(
            "fallback_name",
            markers={"agent_scenario": FakeMarker(kwargs={"name": "alias-flow"})},
        ),
    )

    fixture_gen = plugin.ac_recorder.__wrapped__(request)
    recorder = next(fixture_gen)
    _finish_fixture(fixture_gen)

    assert recorder.run.metadata.scenario == "alias-flow"
    assert list(tmp_path.iterdir()) == []


def test_ac_recorder_falls_back_to_test_name_and_reports_save_errors(
    monkeypatch, tmp_path: Path
) -> None:
    request = FakeRequest(
        config=FakeConfig(**{"--ac-record": True, "--ac-scenarios": str(tmp_path)}),
        node=FakeNode("fallback_name"),
    )

    def broken_save(self: Recorder, path: Path) -> Path:
        raise OSError("permission denied")

    monkeypatch.setattr(plugin.Recorder, "save", broken_save)
    fixture_gen = plugin.ac_recorder.__wrapped__(request)
    recorder = next(fixture_gen)

    assert recorder.run.metadata.scenario == "fallback_name"
    with pytest.raises(pytest.fail.Exception, match="Failed to save cassette"):
        next(fixture_gen)


def test_ac_replay_engine_returns_none_outside_replay_mode() -> None:
    request = FakeRequest(
        config=FakeConfig(**{"--ac-replay": False, "--ac-scenarios": None}),
        node=FakeNode("refund-flow"),
    )

    assert plugin.ac_replay_engine.__wrapped__(request) is None


def test_ac_replay_engine_skips_when_cassette_is_missing(tmp_path: Path) -> None:
    request = FakeRequest(
        config=FakeConfig(**{"--ac-replay": True, "--ac-scenarios": str(tmp_path)}),
        node=FakeNode("missing-flow"),
    )

    with pytest.raises(pytest.skip.Exception, match="No cassette found"):
        plugin.ac_replay_engine.__wrapped__(request)


def test_ac_replay_engine_reports_load_failures(monkeypatch, tmp_path: Path) -> None:
    cassette = tmp_path / "broken-flow.agentrun.json"
    cassette.write_text("{}")
    request = FakeRequest(
        config=FakeConfig(**{"--ac-replay": True, "--ac-scenarios": str(tmp_path)}),
        node=FakeNode("broken-flow"),
    )
    monkeypatch.setattr(
        plugin,
        "load_run",
        lambda path: (_ for _ in ()).throw(ValueError("bad data")),
    )

    with pytest.raises(pytest.fail.Exception, match="Failed to load cassette"):
        plugin.ac_replay_engine.__wrapped__(request)


def test_ac_replay_engine_loads_matching_cassette(tmp_path: Path) -> None:
    _write_cassette(tmp_path / "replay-flow.agentrun.json", scenario="replay-flow")
    request = FakeRequest(
        config=FakeConfig(**{"--ac-replay": True, "--ac-scenarios": str(tmp_path)}),
        node=FakeNode("replay-flow"),
    )

    engine = plugin.ac_replay_engine.__wrapped__(request)

    assert engine is not None
    assert engine.recorded_run.metadata.scenario == "replay-flow"


def test_ac_assert_returns_assertion_engine() -> None:
    engine = plugin.ac_assert.__wrapped__(AgentContractConfig())

    assert isinstance(engine, AssertionEngine)


def test_ac_check_contract_merges_defaults_overrides_and_extra_assertions() -> None:
    config = AgentContractConfig(
        default_assertions=[AssertionSpec(type="contains", target="final_response", value="hello")],
        overrides={
            "refund-flow": ScenarioOverride(
                assertions=[
                    AssertionSpec(type="exact", target="final_response", value="hello world")
                ]
            )
        },
    )
    assertion_engine = MagicMock()
    expected = MagicMock()
    assertion_engine.check.return_value = expected
    run = AgentRun()
    run.metadata.scenario = "refund-flow"

    check = plugin.ac_check_contract.__wrapped__(config, assertion_engine)
    extra = [AssertionSpec(type="regex", target="final_response", value="world")]

    result = check(run, extra)

    assert result is expected
    assertion_engine.check.assert_called_once_with(
        run,
        assertions=[
            config.default_assertions[0],
            config.overrides["refund-flow"].assertions[0],
            extra[0],
        ],
        policies=config.policies,
    )
