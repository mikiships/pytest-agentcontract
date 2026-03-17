"""Tests for the pytest plugin helpers and fixtures."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentcontract.plugin as plugin_module
from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import (
    AgentContractConfig,
    AssertionSpec,
    PolicySpec,
    ScenarioOverride,
)
from agentcontract.plugin import (
    ac_assert,
    ac_check_contract,
    ac_config,
    ac_mode,
    ac_recorder,
    ac_replay_engine,
    pytest_addoption,
    pytest_configure,
)
from agentcontract.replay.engine import ReplayEngine
from agentcontract.types import AgentRun, RunMetadata


class _DummyGroup:
    def __init__(self) -> None:
        self.options: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def addoption(self, *args: object, **kwargs: object) -> None:
        self.options.append((args, kwargs))


class _DummyParser:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.group = _DummyGroup()

    def getgroup(self, name: str, description: str) -> _DummyGroup:
        self.calls.append((name, description))
        return self.group


class _DummyConfig:
    def __init__(self, options: dict[str, object] | None = None) -> None:
        self._options = options or {}
        self.lines: list[tuple[str, str]] = []

    def getoption(self, name: str) -> object:
        return self._options.get(name)

    def addinivalue_line(self, name: str, value: str) -> None:
        self.lines.append((name, value))


class _DummyNode:
    def __init__(self, name: str, markers: dict[str, object] | None = None) -> None:
        self.name = name
        self._markers = markers or {}

    def get_closest_marker(self, name: str) -> object | None:
        return self._markers.get(name)


def _marker(*args: object, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(args=args, kwargs=kwargs)


def _request(
    *,
    options: dict[str, object] | None = None,
    agentcontract_marker: object | None = None,
    agent_scenario_marker: object | None = None,
    node_name: str = "test_node",
) -> SimpleNamespace:
    markers: dict[str, object] = {}
    if agentcontract_marker is not None:
        markers["agentcontract"] = agentcontract_marker
    if agent_scenario_marker is not None:
        markers["agent_scenario"] = agent_scenario_marker
    return SimpleNamespace(
        config=_DummyConfig(options),
        node=_DummyNode(node_name, markers),
    )


def _finish_generator(generator: object) -> None:
    with pytest.raises(StopIteration):
        next(generator)  # type: ignore[arg-type]


def test_plugin_module_reloads_cleanly() -> None:
    reloaded = importlib.reload(plugin_module)

    assert reloaded.ac_mode is not None
    assert reloaded.ac_replay_engine is not None


def test_pytest_addoption_registers_all_agentcontract_options() -> None:
    parser = _DummyParser()

    pytest_addoption(parser)

    assert parser.calls == [("agentcontract", "Agent trajectory testing")]
    assert [args[0] for args, _ in parser.group.options] == [
        "--ac-record",
        "--ac-replay",
        "--ac-config",
        "--ac-scenarios",
    ]


def test_pytest_configure_registers_markers() -> None:
    config = _DummyConfig()

    pytest_configure(config)

    assert config.lines == [
        (
            "markers",
            "agentcontract(scenario): mark test as an agent contract test with a scenario name",
        ),
        ("markers", "agent_scenario(name, **kwargs): alias for agentcontract marker"),
    ]


def test_ac_config_loads_explicit_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = plugin_module.AgentContractConfig(version="2")
    seen: list[Path] = []

    def fake_from_file(
        cls: type[plugin_module.AgentContractConfig], path: Path
    ) -> plugin_module.AgentContractConfig:
        seen.append(path)
        return expected

    monkeypatch.setattr(plugin_module.AgentContractConfig, "from_file", classmethod(fake_from_file))

    config = plugin_module.ac_config.__wrapped__(  # type: ignore[attr-defined]
        _request(options={"--ac-config": str(tmp_path / "agentcontract.yml")})
    )

    assert config is expected
    assert seen == [tmp_path / "agentcontract.yml"]


def test_ac_config_discovers_when_no_explicit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = plugin_module.AgentContractConfig(version="3")

    def fake_discover(cls: type[plugin_module.AgentContractConfig]) -> plugin_module.AgentContractConfig:
        return expected

    monkeypatch.setattr(plugin_module.AgentContractConfig, "discover", classmethod(fake_discover))

    config = plugin_module.ac_config.__wrapped__(_request())  # type: ignore[attr-defined]

    assert config is expected


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"--ac-record": True, "--ac-replay": False}, "record"),
        ({"--ac-record": False, "--ac-replay": True}, "replay"),
        ({}, "live"),
    ],
)
def test_ac_mode_returns_expected_mode(options: dict[str, object], expected: str) -> None:
    assert ac_mode.__wrapped__(_request(options=options)) == expected  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("request_obj", "expected"),
    [
        (_request(agentcontract_marker=_marker("refund-flow")), "refund-flow"),
        (_request(agent_scenario_marker=_marker(name="alias-flow")), "alias-flow"),
        (_request(node_name="fallback_name"), "fallback_name"),
    ],
)
def test_ac_recorder_resolves_scenario_name(
    request_obj: SimpleNamespace, expected: str
) -> None:
    generator = ac_recorder.__wrapped__(request_obj)  # type: ignore[attr-defined]

    recorder = next(generator)

    assert recorder.run.metadata.scenario == expected
    recorder.add_turn(role="user", content="hello")
    _finish_generator(generator)


def test_ac_recorder_autosaves_when_record_mode_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved_paths: list[Path] = []

    def fake_save(self: object, path: str | Path) -> Path:
        saved_paths.append(Path(path))
        return Path(path)

    monkeypatch.setattr("agentcontract.plugin.Recorder.save", fake_save)

    generator = ac_recorder.__wrapped__(  # type: ignore[attr-defined]
        _request(
            options={"--ac-record": True, "--ac-scenarios": str(tmp_path)},
            agentcontract_marker=_marker("recorded-flow"),
        )
    )

    recorder = next(generator)
    recorder.add_turn(role="assistant", content="done")
    _finish_generator(generator)

    assert saved_paths == [tmp_path / "recorded-flow.agentrun.json"]


def test_ac_recorder_uses_default_scenarios_dir_when_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_paths: list[Path] = []

    def fake_save(self: object, path: str | Path) -> Path:
        saved_paths.append(Path(path))
        return Path(path)

    monkeypatch.setattr("agentcontract.plugin.Recorder.save", fake_save)

    generator = ac_recorder.__wrapped__(  # type: ignore[attr-defined]
        _request(
            options={"--ac-record": True},
            agent_scenario_marker=_marker(name="default-dir"),
        )
    )

    next(generator)
    _finish_generator(generator)

    assert saved_paths == [Path("tests/scenarios/default-dir.agentrun.json")]


def test_ac_recorder_fails_when_autosave_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_save(self: object, path: str | Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr("agentcontract.plugin.Recorder.save", fake_save)

    generator = ac_recorder.__wrapped__(  # type: ignore[attr-defined]
        _request(
            options={"--ac-record": True},
            agentcontract_marker=_marker("broken-save"),
        )
    )

    next(generator)

    with pytest.raises(pytest.fail.Exception, match="Failed to save cassette"):
        next(generator)


def test_ac_replay_engine_returns_none_outside_replay_mode() -> None:
    assert ac_replay_engine.__wrapped__(_request()) is None  # type: ignore[attr-defined]


def test_ac_replay_engine_skips_when_cassette_is_missing(tmp_path: Path) -> None:
    request_obj = _request(
        options={"--ac-replay": True, "--ac-scenarios": str(tmp_path)},
        agentcontract_marker=_marker("missing"),
    )

    with pytest.raises(pytest.skip.Exception, match="No cassette found"):
        ac_replay_engine.__wrapped__(request_obj)  # type: ignore[attr-defined]


def test_ac_replay_engine_loads_existing_cassette(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cassette = tmp_path / "alias-flow.agentrun.json"
    cassette.write_text("{}")
    run = AgentRun(metadata=RunMetadata(scenario="alias-flow"))
    seen: list[Path] = []

    def fake_load_run(path: Path) -> AgentRun:
        seen.append(path)
        return run

    monkeypatch.setattr("agentcontract.plugin.load_run", fake_load_run)

    engine = ac_replay_engine.__wrapped__(  # type: ignore[attr-defined]
        _request(
            options={"--ac-replay": True, "--ac-scenarios": str(tmp_path)},
            agent_scenario_marker=_marker(name="alias-flow"),
        )
    )

    assert isinstance(engine, ReplayEngine)
    assert engine.recorded_run is run
    assert seen == [cassette]


def test_ac_replay_engine_fails_when_loading_cassette_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cassette = tmp_path / "bad.agentrun.json"
    cassette.write_text("{}")

    def fake_load_run(path: Path) -> AgentRun:
        raise ValueError("bad cassette")

    monkeypatch.setattr("agentcontract.plugin.load_run", fake_load_run)

    with pytest.raises(pytest.fail.Exception, match="Failed to load cassette"):
        ac_replay_engine.__wrapped__(  # type: ignore[attr-defined]
            _request(
                options={"--ac-replay": True, "--ac-scenarios": str(tmp_path)},
                agentcontract_marker=_marker("bad"),
            )
        )


def test_ac_assert_returns_assertion_engine() -> None:
    assert isinstance(
        ac_assert.__wrapped__(plugin_module.AgentContractConfig()),  # type: ignore[attr-defined]
        AssertionEngine,
    )


def test_ac_check_contract_merges_defaults_overrides_and_extras() -> None:
    default_assertion = AssertionSpec(type="contains", target="final_response", value="default")
    override_assertion = AssertionSpec(type="regex", target="final_response", value="override")
    extra_assertion = AssertionSpec(type="exact", target="final_response", value="extra")
    policies = [PolicySpec(name="allowed", type="tool_allowlist", tools=["lookup_order"])]
    config = AgentContractConfig(
        default_assertions=[default_assertion],
        overrides={"matching-scenario": ScenarioOverride(assertions=[override_assertion])},
        policies=policies,
    )
    calls: list[tuple[AgentRun, list[object], list[PolicySpec]]] = []

    class _DummyAssertionEngine:
        def check(
            self,
            run: AgentRun,
            *,
            assertions: list[object],
            policies: list[PolicySpec],
        ) -> str:
            calls.append((run, assertions, policies))
            return "checked"

    checker = ac_check_contract.__wrapped__(config, _DummyAssertionEngine())  # type: ignore[attr-defined]
    matching_run = AgentRun(metadata=RunMetadata(scenario="matching-scenario"))
    other_run = AgentRun(metadata=RunMetadata(scenario="other-scenario"))

    assert checker(matching_run, extra_assertions=[extra_assertion]) == "checked"
    assert calls[0][1] == [default_assertion, override_assertion, extra_assertion]
    assert calls[0][2] == policies

    assert checker(other_run) == "checked"
    assert calls[1][1] == [default_assertion]
    assert calls[1][1] is not config.default_assertions
