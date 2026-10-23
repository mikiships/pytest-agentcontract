"""Offline integration tests exercising the installed pytest plugin."""

import pytest

from agentcontract.serialization import load_run, save_run
from agentcontract.types import AgentRun, TurnRole

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def isolated_plugin(pytester, monkeypatch):
    # Load exactly once, independently of installed third-party entry points.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest('pytest_plugins = ["agentcontract.plugin"]')


def run_pytest(pytester, *args):
    # In-process runs remain inside the outer coverage measurement.
    return pytester.runpytest("--strict-markers", "-q", *args)


@pytest.mark.parametrize(
    ("flags", "mode"),
    [
        ((), "live"),
        (("--ac-record",), "record"),
        (("--ac-replay",), "replay"),
        (("--ac-record", "--ac-replay"), "record"),
    ],
)
def test_modes(pytester, flags, mode):
    pytester.makepyfile(f"""
        def test_mode(ac_mode, pytestconfig):
            assert ac_mode == {mode!r}
            plugins = [p for p in pytestconfig.pluginmanager.get_plugins()
                       if getattr(p, '__name__', '') == 'agentcontract.plugin']
            assert len(plugins) == 1
    """)
    run_pytest(pytester, *flags).assert_outcomes(passed=1)


@pytest.mark.parametrize("explicit", [False, True])
def test_configuration(pytester, explicit):
    pytester.makefile(".yml", agentcontract="version: discovered")
    path = pytester.makefile(".yml", custom="version: explicit")
    expected = "explicit" if explicit else "discovered"
    pytester.makepyfile(f"""
        def test_config(ac_config):
            assert ac_config.version == {expected!r}
    """)
    run_pytest(pytester, *(("--ac-config", str(path)) if explicit else ())).assert_outcomes(
        passed=1
    )


SCENARIOS = [
    ('@pytest.mark.agentcontract("chosen")', "chosen"),
    ('@pytest.mark.agent_scenario("alias")', "alias"),
    ('@pytest.mark.agentcontract(name="keyword")', "keyword"),
    ('@pytest.mark.agent_scenario(name="alias_keyword")', "alias_keyword"),
    ("@pytest.mark.agentcontract", "test_flow"),
    ("", "test_flow"),
]


@pytest.mark.parametrize(("marker", "scenario"), SCENARIOS)
@pytest.mark.parametrize("override", [False, True])
def test_record_and_replay(pytester, marker, scenario, override):
    directory = pytester.path / ("custom" if override else "tests/scenarios")
    options = ("--ac-scenarios", str(directory)) if override else ()
    pytester.makepyfile(f"""
        import pytest
        {marker}
        def test_flow(ac_recorder, ac_replay_engine, ac_mode):
            assert ac_recorder.run.metadata.scenario == {scenario!r}
            if ac_mode == 'replay':
                assert ac_replay_engine.recorded_run.metadata.scenario == {scenario!r}
                assert ac_replay_engine.recorded_run.turns[0].content == 'offline answer'
                assert ac_replay_engine.finish().ok
            else:
                assert ac_replay_engine is None
            ac_recorder.add_turn('assistant', 'offline answer')
    """)
    run_pytest(pytester, *options).assert_outcomes(passed=1)
    assert not directory.exists()
    run_pytest(pytester, "--ac-record", *options).assert_outcomes(passed=1)
    path = directory / f"{scenario}.agentrun.json"
    run = load_run(path)
    assert run.metadata.scenario == scenario
    assert run.summary.total_turns == 1
    assert run.turns[0].role == TurnRole.ASSISTANT
    assert run.turns[0].content == "offline answer"
    original = path.read_bytes()
    run_pytest(pytester, "--ac-replay", *options).assert_outcomes(passed=1)
    assert path.read_bytes() == original


def test_missing_cassette_skips(pytester):
    pytester.makepyfile("def test_missing(ac_replay_engine): assert False")
    result = run_pytest(pytester, "--ac-replay", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(
        ["*No cassette found at tests/scenarios/test_missing.agentrun.json*"]
    )


def test_malformed_cassette_fails_setup(pytester):
    directory = pytester.path / "tests/scenarios"
    directory.mkdir(parents=True)
    (directory / "test_bad.agentrun.json").write_text("{broken", encoding="utf-8")
    pytester.makepyfile("def test_bad(ac_replay_engine): assert False")
    result = run_pytest(pytester, "--ac-replay")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        ["*Failed to load cassette*test_bad.agentrun.json*JSONDecodeError*"]
    )


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("error", ["OSError", "ValueError", "TypeError"])
def test_io_errors(pytester, operation, error):
    target = "Recorder.save" if operation == "save" else "load_run"
    pytester.makeconftest(f"""
        import pytest
        import agentcontract.plugin as plugin
        pytest_plugins = ['agentcontract.plugin']

        @pytest.fixture(autouse=True)
        def broken_io():
            # Own patch context outlives dependent recorder teardown.
            with pytest.MonkeyPatch.context() as patch:
                def broken(*args, **kwargs):
                    raise {error}('synthetic failure')
                patch.setattr('agentcontract.plugin.{target}', broken)
                yield
    """)
    save_run(AgentRun(), pytester.path / "tests/scenarios/test_io.agentrun.json")
    fixture = "ac_recorder" if operation == "save" else "ac_replay_engine"
    pytester.makepyfile(f"def test_io({fixture}): pass")
    result = run_pytest(pytester, "--ac-record" if operation == "save" else "--ac-replay")
    result.assert_outcomes(passed=int(operation == "save"), errors=1)
    result.stdout.fnmatch_lines(
        [
            f"*Failed to {operation} cassette*test_io.agentrun.json*{error}*synthetic failure*",
        ]
    )
    assert ("teardown" if operation == "save" else "setup") in result.stdout.str()


@pytest.mark.parametrize("scenario", ["special", "ordinary"])
def test_contract_merging_is_repeatable(pytester, scenario):
    pytester.makefile(
        ".yml",
        agentcontract="""
        defaults:
          assertions:
            - {type: contains, target: final_response, value: answer}
        overrides:
          special:
            assertions:
              - {type: exact, target: final_response, value: 'offline answer'}
        policies:
          - {name: safe, type: tool_allowlist, tools: []}
    """,
    )
    pytester.makepyfile(f"""
        from copy import deepcopy
        from agentcontract.config import AssertionSpec
        from agentcontract.types import AgentRun, RunMetadata, Turn, TurnRole, ToolCall

        def test_contract(ac_check_contract, ac_config):
            before = deepcopy(ac_config)
            run = AgentRun(metadata=RunMetadata(scenario={scenario!r}),
                           turns=[Turn(index=0, role=TurnRole.ASSISTANT,
                                       content='offline answer')])
            count = {3 if scenario == "special" else 2}
            for _ in range(2):
                result = ac_check_contract(run)
                assert result.passed
                assert len(result.results) == count
                extra = [AssertionSpec(type='exact', target='final_response', value='wrong')]
                result = ac_check_contract(run, extra_assertions=extra)
                assert len(result.results) == count + 1
                assert result.failed_count == 1
                assert 'Expected exact' in result.failures()[0].message
                assert len(extra) == 1
            run.turns[0].tool_calls = [ToolCall(id='1', function='unsafe', arguments={{}})]
            result = ac_check_contract(run)
            assert result.failed_count == 1
            assert 'Disallowed tools' in result.failures()[0].message
            assert ac_config == before
    """)
    run_pytest(pytester).assert_outcomes(passed=1)


def test_contract_with_default_configuration(pytester):
    pytester.makepyfile("""
        from agentcontract.types import AgentRun
        def test_defaults(ac_check_contract, ac_config):
            assert ac_config.default_assertions == []
            result = ac_check_contract(AgentRun())
            assert result.passed
            assert result.results == []
    """)
    run_pytest(pytester).assert_outcomes(passed=1)
