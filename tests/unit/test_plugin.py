"""Offline pytest projects exercising the public plugin fixture lifecycle."""

import pytest

from agentcontract.serialization import load_run

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def isolated_plugins(monkeypatch):
    # Load exactly once explicitly, independent of installed entry points.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def run_project(pytester, *args):
    # In-process execution lets the outer coverage collector observe fixtures.
    return pytester.runpytest_inprocess("-p", "agentcontract.plugin", "--strict-markers", *args)


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
    pytester.makepyfile(f"def test_mode(ac_mode):\n    assert ac_mode == {mode!r}")
    run_project(pytester, *flags).assert_outcomes(passed=1)


@pytest.mark.parametrize("explicit", [False, True])
def test_configuration(pytester, explicit):
    pytester.makefile(".yml", agentcontract="version: discovered")
    config = pytester.makefile(".yml", selected="version: explicit")
    expected = "explicit" if explicit else "discovered"
    pytester.makepyfile(
        f"def test_config(ac_config):\n    assert ac_config.version == {expected!r}"
    )
    run_project(pytester, *(["--ac-config", str(config)] if explicit else [])).assert_outcomes(
        passed=1
    )


@pytest.mark.parametrize(
    ("marker", "scenario"),
    [
        ('@pytest.mark.agentcontract("positional")', "positional"),
        ('@pytest.mark.agentcontract(name="keyword")', "keyword"),
        ('@pytest.mark.agent_scenario("alias")', "alias"),
        ('@pytest.mark.agent_scenario(name="alias_keyword")', "alias_keyword"),
        ("", "test_flow"),
    ],
)
@pytest.mark.parametrize("override", [False, True])
def test_record_and_replay(pytester, marker, scenario, override):
    directory = "custom" if override else "tests/scenarios"
    flags = ["--ac-scenarios", directory] if override else []
    pytester.makepyfile(f"""
        import pytest
        {marker}
        def test_flow(ac_recorder, ac_replay_engine, ac_mode):
            assert ac_recorder.run.metadata.scenario == {scenario!r}
            if ac_mode == "record":
                assert ac_replay_engine is None
                ac_recorder.add_turn(role="user", content="question")
                ac_recorder.add_turn(role="assistant", content="answer",
                    tool_calls=[{{"function": "lookup", "arguments": {{"id": 1}},
                                  "result": "found"}}])
            else:
                assert ac_replay_engine.recorded_run.metadata.scenario == {scenario!r}
                assert ac_replay_engine.tool_stub.get_result("lookup", {{"id": 1}}) == "found"
                assert ac_replay_engine.finish().ok
    """)
    run_project(pytester, "--ac-record", *flags).assert_outcomes(passed=1)
    path = pytester.path / directory / f"{scenario}.agentrun.json"
    run = load_run(path)
    assert [turn.content for turn in run.turns] == ["question", "answer"]
    assert run.summary.total_turns == 2
    original = path.read_bytes()
    run_project(pytester, "--ac-replay", *flags).assert_outcomes(passed=1)
    assert path.read_bytes() == original  # Replay's empty recorder must not overwrite it.


@pytest.mark.parametrize("flags", [(), ("--ac-replay",)])
def test_no_recording_outside_record_mode(pytester, flags):
    pytester.makepyfile("""
        def test_live(ac_recorder):
            ac_recorder.add_turn(role="assistant", content="unsaved")
    """)
    run_project(pytester, *flags).assert_outcomes(passed=1)
    assert not list(pytester.path.rglob("*.agentrun.json"))


def test_missing_cassette_skips(pytester):
    pytester.makepyfile("""
        def test_absent(ac_replay_engine):
            assert False, "missing cassette must skip before the test body"
    """)
    result = run_project(pytester, "--ac-replay", "-rs")
    result.assert_outcomes(skipped=1)
    assert "No cassette found at tests/scenarios/test_absent.agentrun.json" in result.stdout.str()


def test_malformed_cassette_fails(pytester):
    path = pytester.path / "tests/scenarios/test_bad.agentrun.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    pytester.makepyfile("def test_bad(ac_replay_engine):\n    assert False")
    result = run_project(pytester, "--ac-replay")
    result.assert_outcomes(errors=1)
    assert "Failed to load cassette 'tests/scenarios/test_bad.agentrun.json'" in result.stdout.str()
    assert "JSONDecodeError" in result.stdout.str()


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("exception", ["OSError", "ValueError", "TypeError"])
def test_cassette_errors(pytester, operation, exception):
    target = "plugin.Recorder, 'save'" if operation == "save" else "plugin, 'load_run'"
    pytester.makeconftest(f"""
        import pytest
        import agentcontract.plugin as plugin
        @pytest.fixture(autouse=True)
        def inject_error(monkeypatch):
            def broken(*args, **kwargs):
                raise {exception}("injected boundary error")
            monkeypatch.setattr({target}, broken)
    """)
    path = pytester.path / "cassettes/test_boundary.agentrun.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")
    fixture = "ac_recorder" if operation == "save" else "ac_replay_engine"
    pytester.makepyfile(f"def test_boundary({fixture}):\n    pass")
    flag = "--ac-record" if operation == "save" else "--ac-replay"
    result = run_project(pytester, flag, "--ac-scenarios", "cassettes")
    result.assert_outcomes(errors=1, passed=1 if operation == "save" else 0)
    assert (
        f"Failed to {operation} cassette 'cassettes/test_boundary.agentrun.json' "
        f"({exception}): injected boundary error"
    ) in result.stdout.str()


def test_contract_composition_and_configuration_immutability(pytester):
    pytester.makefile(
        ".yml",
        agentcontract="""
        defaults:
          assertions:
            - {type: contains, target: final_response, value: answer}
        overrides:
          matching:
            assertions:
              - {type: exact, target: final_response, value: answer}
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
            recorder = Recorder(scenario="matching")
            recorder.add_turn(role="assistant", content="answer")
            extra = AssertionSpec(type="regex", target="final_response", value="^answer$")
            for _ in range(2):
                result = ac_check_contract(recorder.run, [extra])
                assert result.passed
                assert [r.assertion.type for r in result.results] == [
                    "contains", "exact", "regex", "policy:safe"]
            failed = ac_check_contract(recorder.run, [
                AssertionSpec(type="exact", target="final_response", value="wrong")])
            assert not failed.passed and failed.failed_count == 1
            assert "wrong" in failed.failures()[0].message
            recorder.run.turns[-1].content = "answer extended"
            failed = ac_check_contract(recorder.run)
            assert not failed.passed and failed.failed_count == 1
            assert failed.failures()[0].assertion.type == "exact"
            recorder.run.metadata.scenario = "unmatched"
            recorder.run.turns[-1].content = "missing"
            failed = ac_check_contract(recorder.run)
            assert not failed.passed and failed.failed_count == 1
            assert failed.failures()[0].assertion.type == "contains"
            recorder.run.turns[-1].content = "answer"
            result = ac_check_contract(recorder.run)
            assert result.passed and len(result.results) == 2
            recorder.add_turn(role="assistant", content="answer",
                              tool_calls=[{"function": "dangerous"}])
            result = ac_check_contract(recorder.run, [])
            assert not result.passed and result.failed_count == 1
            assert "dangerous" in result.failures()[0].message
            assert ac_config == original
    """)
    run_project(pytester).assert_outcomes(passed=1)
