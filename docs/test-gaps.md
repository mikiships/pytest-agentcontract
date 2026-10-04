# Test gap report

## Reproducible baseline

Measured on 2026-10-04 at `origin/main` revision
`17c52033d3cadefe7c1151b6c76c74995b0db71a`, in an isolated worktree.
This report identifies gaps; it adds no tests and claims no confirmed production defects.

Environment: macOS, CPython 3.12.14, pytest 9.0.3, pytest-cov 7.1.0,
coverage.py 7.14.0, PyYAML 6.0.3, jsonschema 4.26.0. Install the project's
existing development extra in a fresh environment (no lockfile modification):

```sh
uv venv --python 3.12 /tmp/test-gap-20261004-venv
uv pip install --python /tmp/test-gap-20261004-venv/bin/python -e '.[dev]'
source /tmp/test-gap-20261004-venv/bin/activate
python -m pytest --collect-only -q
COVERAGE_FILE=/tmp/test-gap-20261004.coverage python -m pytest tests --cov=agentcontract --cov-branch --cov-report=term-missing --cov-report=json:/tmp/test-gap-20261004.json
```

Default collection finds **77 tests**, all under `tests/unit/`; the measured run
has **77 passed, zero skipped or failed**. The requested command reports
728/1,351 statements and 286/484 branch exits covered (55.26% combined).
It emits `module-not-measured` because pytest loads the installed plugin before
pytest-cov starts; import-time lines are therefore incorrectly presented as gaps.
Use this startup measurement to interpret missing code:

```sh
COVERAGE_FILE=/tmp/test-gap-20261004-startup.coverage python -m coverage run --branch --source=agentcontract -m pytest tests
COVERAGE_FILE=/tmp/test-gap-20261004-startup.coverage python -m coverage json -o /tmp/test-gap-20261004-startup.json
COVERAGE_FILE=/tmp/test-gap-20261004-startup.coverage python -m coverage report -m
```

The startup run also has **77 passed**, with no coverage warning:
**1,006/1,351 statements (74.46%)**, **286/484 branch exits (59.09%)**,
**70.41% combined coverage**, and 88 partially covered branches.
The combined percentage is not the branch percentage. Both reports include all
17 Python files under `src/agentcontract`, including unexecuted code; the JSON
file inventory was checked against that directory. There are no excluded lines.
Generated coverage artifacts remain outside the repository.

`.github/workflows/ci.yml` installs `.[dev]`, runs `ruff check src/ tests/`,
`pytest tests/ -v --tb=short`, and a non-blocking mypy check on Python 3.10–3.13.
CI does not measure coverage. This baseline runs only Python 3.12, not the full
CI matrix; lint and typing are not revalidated by this documentation-only change.
Dependency resolution is unpinned, as in CI; the versions above identify this run.

`pyproject.toml` sets `testpaths = ["tests"]`. The three fixture consumers in
`examples/customer_support/test_support.py` are outside both default discovery
and CI's explicit `tests/` selection. They were inspected but not executed and
are not counted as automated coverage of plugin behavior.

## Module inventory

Counts below come from the startup JSON. Executed lines, imports, and callable
checks do not establish behavioral assertions.

| Module under `src/agentcontract/` | Statements covered/total | Branch exits covered/total | Existing test evidence |
| --- | ---: | ---: | --- |
| `__init__.py` | 3/9 | 0/2 | No dedicated package-export test |
| `adapters/__init__.py` | 10/10 | 2/2 | `test_adapters.py`: lazy import/callable checks |
| `adapters/langgraph.py` | 68/88 | 25/42 | `test_adapters.py`: graph wrapping and extraction |
| `adapters/llamaindex.py` | 61/82 | 22/38 | `test_adapters.py`: chat/query and source extraction |
| `adapters/openai_agents.py` | 46/146 | 18/80 | `test_adapters.py`: extraction helpers; no runner invocation |
| `assertions/__init__.py` | 2/2 | 0/0 | Imports only |
| `assertions/engine.py` | 151/185 | 54/82 | `test_assertions.py` (22 tests); config policy check |
| `cli.py` | 26/68 | 3/14 | `test_cli.py` (1 invalid-info test) |
| `config.py` | 134/148 | 32/42 | `test_config.py` (7 tests) |
| `plugin.py` | 32/87 | 0/24 | Startup hooks only; no fixture consumers in `tests/` |
| `recorder/__init__.py` | 2/2 | 0/0 | Imports only |
| `recorder/core.py` | 72/78 | 14/16 | `test_recorder.py` (5 tests), adapters/interceptors |
| `recorder/interceptors.py` | 125/135 | 39/48 | `test_interceptors.py` (12 tests, fake SDKs) |
| `replay/__init__.py` | 2/2 | 0/0 | Imports only |
| `replay/engine.py` | 89/107 | 30/42 | `test_replay.py` (8 tests) |
| `serialization.py` | 116/131 | 47/52 | `test_serialization.py` (6 tests), recorder round trip, CLI error |
| `types.py` | 67/71 | 0/0 | Indirect construction; no dedicated test file |

## Prioritized findings

All five are measured execution gaps corroborated by source and test inspection.
Priority reflects impact and regression risk, not simply the lowest percentage.
Source line ranges refer to the baseline revision above.

| Rank | Gap and impact | Source location / measurement | Existing-test evidence |
| --- | --- | --- | --- |
| 1 — High | Recorder/replay fixture lifecycle: cassette creation, selection, and error reporting can regress without any CI test failing. | `plugin.py:78–146`; fixture bodies unexecuted, plugin has 0/24 branch exits covered. | `test_recorder_save_load` tests direct persistence, not pytest teardown; replay tests construct engines directly. No `ac_recorder` or `ac_replay_engine` consumer under `tests/`. |
| 2 — High | Contract fixture assembly: defaults, scenario assertions, extras, or policies could be silently omitted. | `plugin.py:149–187`; all fixture logic unexecuted. | `test_config_from_dict` checks parsing; `test_assertions.py` invokes the engine directly. Neither exercises `ac_check_contract`. |
| 3 — High | Mode/configuration selection: option parsing and precedence have no behavioral assertions, risking incorrect live/record/replay selection. | `plugin.py:19–74`; hooks execute at startup, but `61–64` and `70–74` do not. | `test_discover_accepts_file_path_start` covers direct discovery, not `--ac-config`; no `ac_mode` or `ac_config` consumers in collected tests. |
| 4 — High | OpenAI Agents runner interception: public wrapper, async/sync forwarding, and restoration are wholly untested. | `adapters/openai_agents.py:26–91`; `38–91` unexecuted; module branch exits 18/80. | `TestOpenAIAgentsAdapter` calls private extractors; `test_lazy_import_record_runner` only asserts callable. |
| 5 — Medium | Replay completion without supplied turns: unused stubs may go undetected if this path regresses. | `replay/engine.py:129–140`; `131–140` unexecuted. | Every existing `finish` test supplies `actual_turns`; `test_tool_stub_has_results` does not validate aggregate completion. |

### 1. Recorder/replay fixture lifecycle

Suggested file: `tests/unit/test_plugin.py`, using pytest's built-in `pytester`
plugin and subprocess runs in temporary projects. Assert pytest outcomes and
cassette contents after teardown, not just that a fixture can be imported.

- Parameterize `@pytest.mark.agentcontract("refund")`,
  `@pytest.mark.agent_scenario("refund")`, `agent_scenario(name="refund")`,
  no marker, and both markers. Expect the primary marker to win, positional
  arguments before `name`, and the test node name as fallback. Include a
  parametrized node to check its resulting filename. Run with `--strict-markers`
  to verify both registrations.
- With `--ac-record`, add a user turn and verify teardown creates
  `tests/scenarios/<scenario>.agentrun.json` with finalized totals. Repeat with
  `--ac-scenarios <temporary-directory>`; without `--ac-record`, expect no save.
  Include a failing test body: pytest still tears down the yielded fixture and
  saves the captured run; assert the original test failure remains visible.
- Monkeypatch `Recorder.save` to raise each of `OSError`, `ValueError`, and
  `TypeError`. Expect a teardown error containing `Failed to save cassette`,
  the path, and exception type. Inject errors rather than relying on permissions.
- Without `--ac-replay`, expect `ac_replay_engine is None`. In replay mode,
  write a valid cassette and verify its scenario and recorded tool result load.
  A missing file must **skip**, not pass or return `None` to the test body
  (the fixture docstring is less precise than the implementation).
- Write malformed JSON and a turn with an invalid role; expect setup errors
  containing `Failed to load cassette`, path, and exception type. Patch
  `load_run` for deterministic `OSError`/`TypeError` cases. These protect
  fixture-level diagnostics beyond the existing CLI malformed-cassette test.

Keep all files and subprocess working directories under `tmp_path`/`pytester`;
use synthetic turns and no API credentials or network. Register the installed
plugin once, avoiding duplicate explicit loading when autoload is enabled.

### 2. Contract assertion merging and policies

Suggested file: `tests/unit/test_plugin.py`.

Supply `ac_config` with a default `contains(final_response, "refund")`, an override
for scenario `refund` using `called_count(tool:lookup_order, "1")`, and an extra
`exact(final_response, "refund complete")`. Create a run with that response and
one `lookup_order` call, plus a `tool_allowlist` policy allowing only that tool.
Expect four successful results in default → override → extra → policy order.
Then add a disallowed tool: the policy must fail even while content assertions
pass. A different scenario must omit only the override; absent/empty extras
must add nothing. Invoke the callable twice and verify neither configuration
lists nor caller extras accumulate or mutate. Use the real assertion engine,
temporary config, and constructed runs; no SDK installation is necessary.

### 3. CLI mode and configuration precedence

Suggested file: `tests/unit/test_plugin.py` with isolated pytester projects.

- Assert `--help` exposes all four options. Parameterize no flags → `live`,
  `--ac-record` → `record`, `--ac-replay` → `replay`, and both → `record` for
  `ac_mode`. Also characterize both flags with fixture consumers: recorder and
  replay fixtures inspect their respective flags independently. Unified mode
  enforcement is a possible design improvement, not a demonstrated defect here.
- Put version `"2"` in a discovered parent `agentcontract.yml` and version `"3"`
  in an explicit config; expect `--ac-config` to select `"3"` and no option to
  select `"2"`. For the no-config default, mock discovery's search boundary or
  `from_file`/existence checks so a developer's ancestor config cannot leak in.
- Explicit nonexistent config must produce a fixture setup error, without
  silently falling back. An invalid YAML file should likewise fail setup.
- Verify `--ac-scenarios` selects the cassette directory as in finding 1.
  Current fixtures use that option or `tests/scenarios`, not config include
  globs; do not assert config-derived directory behavior without a design change.

### 4. OpenAI Agents public runner wrapper

Suggested file: `tests/unit/test_adapters.py` (new runner-focused class).

Inject a fake `agents` module into `sys.modules` using `monkeypatch`, with a
`Runner` exposing async `run` and sync `run_sync`. Return a sentinel result with
`final_output="done"`. Invoke both public methods after `record_runner(recorder)`;
use `asyncio.run` for async coverage without a new pytest dependency. Assert
argument forwarding, result identity, one expected recorded turn per call,
latency, and original method restoration after `unpatch()`. Control the clock
and always restore patches in `finally` to prevent cross-test contamination.

Parameterize only-sync and only-async runners: both should work. A runner
with neither supported method should raise `ValueError`, a noncallable method
should raise `TypeError`, and missing SDK import should raise the actionable
`ImportError`. Also test a runner method raising a
sentinel exception: propagate it and record no successful result. Partial
patch cleanup on invalid runner definitions is a hypothesis worth investigating,
not a behavior verified by this baseline.

The development extra does not install OpenAI, Anthropic, LangChain,
LlamaIndex, or OpenAI Agents SDKs; current adapter/interceptor tests use fakes.
`openai-agents` is not included even in the declared `all` extra. Fake-based
wrapper tests need no dependency changes, but cannot prove real SDK version
compatibility. No live-provider or optional-SDK integration claim is made.

### 5. Replay stub-consumption completion

Suggested file: `tests/unit/test_replay.py`.

Construct a run with two calls to `lookup_order` and one to `process_refund`.
Call `finish()` without `actual_turns`: expect `missing_tools == 3`, two
per-function errors, and `ok is False`. Consume one lookup result and repeat:
expect two missing tools. Consume the rest: expect no errors, zero missing,
and `ok is True`. Also test an empty run and repeated `finish()` calls; checking
completion must not consume stubs. Use in-memory `AgentRun`/`ToolCall` objects
and assert exact counts rather than only truthiness.

## Scope of prioritization

Serialization coercion/round trips, recorder totals, direct policy assertions,
config parsing/discovery, CLI, and all adapters were checked against the missing
lines and current tests. CLI validation/init and assertion target resolution
also have uncovered paths; they rank below the integration boundaries above
for this five-finding report. Existing passing tests establish neither full
SDK compatibility nor correctness of every executed branch. Proposed cases
are follow-up work, not additional tests executed as part of this baseline.
