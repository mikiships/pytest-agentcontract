"""Public plugin contracts, exercised through isolated in-process pytest sessions."""

import pytest

from agentcontract.serialization import load_run

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def isolated_plugins(monkeypatch):
    # Explicit loading avoids duplicate entry-point registration. In-process runs
    # keep nested fixture execution inside the outer coverage collector.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def run(pytester, *args):
    return pytester.runpytest_inprocess("-p", "agentcontract.plugin", "--strict-markers", *args)


@pytest.mark.parametrize(
    ("flags", "mode"),
    [
        ([], "live"),
        (["--ac-record"], "record"),
        (["--ac-replay"], "replay"),
        (["--ac-record", "--ac-replay"], "record"),
    ],
)
def test_modes_and_recorder_lifecycle(pytester, flags, mode):
    pytester.makepyfile(
        """
        def test_mode(ac_mode, ac_recorder):
            assert ac_mode == MODE
            ac_recorder.add_turn("assistant", "done", prompt_tokens=2, completion_tokens=3)
    """.replace("MODE", repr(mode))
    )
    result = run(pytester, *flags)
    result.assert_outcomes(passed=1)
    path = pytester.path / "tests/scenarios/test_mode.agentrun.json"
    assert path.exists() == (mode == "record")
    if mode == "record":
        recorded = load_run(path)
        assert recorded.summary.total_turns == 1
        assert recorded.summary.total_tokens.total == 5
        assert recorded.summary.total_duration_ms >= 0


SCENARIOS = [
    ("@pytest.mark.agentcontract('chosen')", "chosen"),
    ("@pytest.mark.agentcontract(name='chosen')", "chosen"),
    ("@pytest.mark.agent_scenario('chosen')", "chosen"),
    ("@pytest.mark.agent_scenario(name='chosen')", "chosen"),
    ("", "test_case"),
]


@pytest.mark.parametrize(("marker", "scenario"), SCENARIOS)
@pytest.mark.parametrize("override", [False, True])
def test_scenario_record_and_replay(pytester, marker, scenario, override):
    directory = pytester.path / ("custom" if override else "tests/scenarios")
    args = ["--ac-scenarios", str(directory)] if override else []
    pytester.makepyfile(
        test_record=f"""
        import pytest
        {marker}
        def test_case(ac_recorder):
            assert ac_recorder.run.metadata.scenario == {scenario!r}
            ac_recorder.add_turn("assistant", "done", tool_calls=[
                {{"function": "lookup", "arguments": {{"id": 1}}, "result": "found"}}
            ])
    """
    )
    run(pytester, "test_record.py", "--ac-record", *args).assert_outcomes(passed=1)
    cassette = directory / f"{scenario}.agentrun.json"
    assert load_run(cassette).metadata.scenario == scenario
    assert list(directory.glob("*.agentrun.json")) == [cassette]
    pytester.makepyfile(
        test_replay=f"""
        import pytest
        {marker}
        def test_case(ac_replay_engine):
            assert ac_replay_engine.recorded_run.metadata.scenario == {scenario!r}
            assert ac_replay_engine.tool_stub.get_result("lookup", {{"id": 1}}) == "found"
            assert ac_replay_engine.finish().ok
    """
    )
    run(pytester, "test_replay.py", "--ac-replay", *args).assert_outcomes(passed=1)


@pytest.mark.parametrize("explicit", [False, True])
def test_config_loading(pytester, explicit):
    pytester.makefile(".yml", agentcontract="replay:\n  model: discovered")
    selected = pytester.makefile(".yml", selected="replay:\n  model: explicit")
    expected = "explicit" if explicit else "discovered"
    pytester.makepyfile(f"""
        def test_config(ac_config):
            assert ac_config.replay.model == {expected!r}
    """)
    args = ["--ac-config", str(selected)] if explicit else []
    run(pytester, *args).assert_outcomes(passed=1)


def test_cli_help(pytester):
    result = run(pytester, "--help")
    assert result.ret == 0
    for option in ("--ac-record", "--ac-replay", "--ac-config", "--ac-scenarios"):
        assert option in result.stdout.str()


def test_replay_disabled(pytester):
    pytester.makepyfile("""
        def test_disabled(ac_replay_engine):
            assert ac_replay_engine is None
    """)
    run(pytester).assert_outcomes(passed=1)


@pytest.mark.parametrize("kind", ["missing", "malformed", "unreadable"])
def test_replay_errors(pytester, kind):
    directory = pytester.path / "cassettes"
    directory.mkdir()
    path = directory / "test_case.agentrun.json"
    if kind == "malformed":
        path.write_text("{invalid", encoding="utf-8")
    elif kind == "unreadable":
        # A directory is reliably unreadable as a file, even under root.
        path.mkdir()
    pytester.makepyfile("""
        def test_case(ac_replay_engine):
            pytest.fail("setup should skip or fail before the test")
        import pytest
    """)
    result = run(pytester, "--ac-replay", "--ac-scenarios", str(directory), "-rs")
    result.assert_outcomes(**({"skipped": 1} if kind == "missing" else {"errors": 1}))
    output = result.stdout.str()
    assert str(path) in output
    assert ("No cassette found" if kind == "missing" else "Failed to load cassette") in output
    if kind != "missing":
        assert ("JSONDecodeError" if kind == "malformed" else "IsADirectoryError") in output


@pytest.mark.parametrize("exception", ["OSError", "ValueError", "TypeError"])
def test_save_failure_is_teardown_error(pytester, exception):
    pytester.makeconftest(f"""
        from agentcontract.recorder.core import Recorder
        import pytest

        @pytest.fixture(autouse=True)
        def fail_save(monkeypatch):
            def save(self, path):
                raise {exception}("synthetic save failure")
            monkeypatch.setattr(Recorder, "save", save)
    """)
    pytester.makepyfile("""
        def test_case(ac_recorder):
            ac_recorder.add_turn("assistant", "done")
    """)
    result = run(pytester, "--ac-record")
    result.assert_outcomes(passed=1, errors=1)
    output = result.stdout.str()
    assert "ERROR at teardown" in output
    assert "tests/scenarios/test_case.agentrun.json" in output
    assert f"({exception}): synthetic save failure" in output


def test_contract_merging_policy_and_call_isolation(pytester):
    pytester.makefile(
        ".yml",
        agentcontract="""
defaults:
  assertions:
    - {type: contains, target: final_response, value: hello}
overrides:
  selected:
    assertions:
      - {type: exact, target: final_response, value: hello world}
policies:
  - {name: safe, type: tool_allowlist, tools: [lookup]}
""",
    )
    pytester.makepyfile("""
        from copy import deepcopy
        from agentcontract.config import AssertionSpec
        from agentcontract.recorder.core import Recorder

        def test_contract(ac_check_contract, ac_config):
            original = deepcopy(ac_config)
            recorder = Recorder(scenario="selected")
            recorder.add_turn("assistant", "hello world")
            run = recorder.run
            result = ac_check_contract(run)
            assert result.passed and len(result.results) == 3
            extra = [AssertionSpec(type="exact", target="final_response", value="wrong")]
            result = ac_check_contract(run, extra_assertions=extra)
            assert result.failed_count == 1 and len(result.results) == 4
            assert len(extra) == 1
            assert ac_check_contract(run).passed
            assert len(ac_check_contract(run).results) == 3
            run.turns[0].content = "hello other"
            assert ac_check_contract(run).failed_count == 1
            run.metadata.scenario = "unmatched"
            result = ac_check_contract(run)
            assert result.passed and len(result.results) == 2
            run.turns[0].content = "wrong"
            assert ac_check_contract(run).failed_count == 1
            run.turns[0].content = "hello"
            recorder.add_turn("assistant", tool_calls=[{"function": "delete"}])
            result = ac_check_contract(run)
            assert result.failed_count == 1
            assert "Disallowed tools" in result.failures()[0].message
            assert ac_config == original
    """)
    run(pytester).assert_outcomes(passed=1)
