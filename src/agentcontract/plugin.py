"""pytest plugin: provides fixtures, markers, and CLI options for agentcontract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import pytest

from agentcontract.assertions.engine import AssertionEngine, ContractResult
from agentcontract.config import AgentContractConfig
from agentcontract.recorder.core import Recorder
from agentcontract.replay.engine import ReplayEngine, ReplayResult
from agentcontract.serialization import load_run, save_run
from agentcontract.types import AgentRun


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add agentcontract CLI options to pytest."""
    group = parser.getgroup("agentcontract", "Agent trajectory testing")
    group.addoption(
        "--ac-record",
        action="store_true",
        default=False,
        help="Record agent trajectories (requires live LLM/tool access)",
    )
    group.addoption(
        "--ac-replay",
        action="store_true",
        default=False,
        help="Replay trajectories from recorded cassettes (deterministic, no network)",
    )
    group.addoption(
        "--ac-config",
        default=None,
        help="Path to agentcontract.yml config file",
    )
    group.addoption(
        "--ac-scenarios",
        default=None,
        help="Path to scenarios directory (overrides config)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "agentcontract(scenario): mark test as an agent contract test with a scenario name",
    )
    config.addinivalue_line(
        "markers",
        "agent_scenario(name, **kwargs): alias for agentcontract marker",
    )


@pytest.fixture
def ac_config(request: pytest.FixtureRequest) -> AgentContractConfig:
    """Provide the parsed agentcontract configuration."""
    config_path = request.config.getoption("--ac-config")
    if config_path:
        return AgentContractConfig.from_file(Path(config_path))
    return AgentContractConfig.discover()


@pytest.fixture
def ac_mode(request: pytest.FixtureRequest) -> str:
    """Return the current mode: 'record', 'replay', or 'live'."""
    if request.config.getoption("--ac-record"):
        return "record"
    if request.config.getoption("--ac-replay"):
        return "replay"
    return "live"


@pytest.fixture
def ac_recorder(request: pytest.FixtureRequest) -> Generator[Recorder, None, None]:
    """Provide a Recorder instance that auto-saves after the test.

    The cassette is saved to tests/scenarios/<scenario>.agentrun.json.
    """
    marker = request.node.get_closest_marker("agentcontract") or request.node.get_closest_marker(
        "agent_scenario"
    )
    scenario = ""
    if marker and marker.args:
        scenario = marker.args[0]
    elif marker and marker.kwargs.get("name"):
        scenario = marker.kwargs["name"]
    else:
        scenario = request.node.name

    recorder = Recorder(scenario=scenario)

    with recorder.recording():
        yield recorder

    # Auto-save if in record mode
    if request.config.getoption("--ac-record"):
        scenarios_dir = request.config.getoption("--ac-scenarios") or "tests/scenarios"
        path = Path(scenarios_dir) / f"{scenario}.agentrun.json"
        recorder.save(path)


@pytest.fixture
def ac_replay_engine(
    request: pytest.FixtureRequest,
) -> ReplayEngine | None:
    """Provide a ReplayEngine loaded from the matching cassette.

    Returns None if not in replay mode or cassette doesn't exist.
    """
    if not request.config.getoption("--ac-replay"):
        return None

    marker = request.node.get_closest_marker("agentcontract") or request.node.get_closest_marker(
        "agent_scenario"
    )
    scenario = ""
    if marker and marker.args:
        scenario = marker.args[0]
    elif marker and marker.kwargs.get("name"):
        scenario = marker.kwargs["name"]
    else:
        scenario = request.node.name

    scenarios_dir = request.config.getoption("--ac-scenarios") or "tests/scenarios"
    cassette_path = Path(scenarios_dir) / f"{scenario}.agentrun.json"

    if not cassette_path.exists():
        pytest.skip(f"No cassette found at {cassette_path}")
        return None

    run = load_run(cassette_path)
    return ReplayEngine(run)


@pytest.fixture
def ac_assert(ac_config: AgentContractConfig) -> AssertionEngine:
    """Provide an AssertionEngine instance."""
    return AssertionEngine()


@pytest.fixture
def ac_check_contract(
    ac_config: AgentContractConfig,
    ac_assert: AssertionEngine,
) -> Any:
    """Return a callable that checks a trajectory against its contract.

    Usage:
        def test_refund(ac_recorder, ac_check_contract):
            # ... record turns ...
            result = ac_check_contract(ac_recorder.run)
            assert result.passed, result.failures()
    """

    def _check(
        run: AgentRun,
        extra_assertions: list[Any] | None = None,
    ) -> ContractResult:
        # Merge default assertions with scenario overrides
        assertions = list(ac_config.default_assertions)
        scenario_name = run.metadata.scenario
        if scenario_name in ac_config.overrides:
            assertions.extend(ac_config.overrides[scenario_name].assertions)
        if extra_assertions:
            assertions.extend(extra_assertions)

        return ac_assert.check(
            run,
            assertions=assertions,
            policies=ac_config.policies,
        )

    return _check
