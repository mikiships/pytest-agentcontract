# Test coverage gaps

Audit date: 2026-10-19. Base: `origin/main` at
`17c52033d3cadefe7c1151b6c76c74995b0db71a`. Scope: offline tests and this report;
no production, dependency, or CI policy changes.

## Environment and method

A separate worktree and fresh virtual environment were used. Python 3.12.14 on
macOS; pytest 9.0.3, pytest-cov 7.1.0, coverage 7.14.0, PyYAML 6.0.3,
jsonschema 4.26.0, ruff 0.15.14, mypy 2.1.0, pluggy 1.6.0.
Installed only the editable project and its `dev` extra; no provider SDKs.
The existing checkout, dirty `uv.lock`, and existing `.coverage` were not used.
Dependencies were resolved for this environment without updating a lockfile.

Baseline: **77 passed, no failures**. Final: **104 passed, no failures**;
focused plugin tests: **27 passed**. The original eight test modules cover
assertions, adapters, CLI, config, interceptors, recorder, replay, and serialization.
There were no fixture-name or `pytester` references in the original tests.

The exact requested measurement (run from the worktree) was:

```sh
COVERAGE_FILE=/tmp/ac-gap-baseline.coverage /tmp/ac-test-gap-env/bin/pytest tests --cov=agentcontract --cov-branch --cov-report=term-missing --cov-report=json:/tmp/ac-gap-baseline.json
```

For the final run, replace both `baseline` filenames with `final`.
That measurement reports combined line/branch coverage of **55.26% → 60.49%**
repository-wide and **1.80% → 76.58%** for the plugin. These are not reliable
import-time coverage totals: pytest loads its installed plugin entry point and
transitive imports before pytest-cov starts collecting. For example, module
class definitions appear unexecuted despite their objects being used by tests.

To capture startup as well, the following commands were run before and after
adding the tests (replace `early-baseline` with `early-final` for the latter):

```sh
COVERAGE_FILE=/tmp/ac-gap-early-baseline.coverage /tmp/ac-test-gap-env/bin/python -m coverage run --branch --source=agentcontract -m pytest tests
COVERAGE_FILE=/tmp/ac-gap-early-baseline.coverage /tmp/ac-test-gap-env/bin/coverage json -o /tmp/ac-gap-early-baseline.json
```

With startup measured, combined coverage is **70.41% → 75.37%** overall and
**28.83% → 99.10%** for the plugin. Plugin line coverage rises from **32/87
(36.78%) to 86/87 (98.85%)**; branch coverage from **0/24 to 24/24 (100%)**.
Coverage percentages measure execution, not the adequacy of behavioral assertions.
No coverage thresholds or exclusions were added.

## Repository-wide measured inventory

Counts below are executed/total statements and branch destinations from the
startup-aware runs. Paths are relative to `src/agentcontract/`.

| Module | Baseline lines | Final lines | Baseline branches | Final branches |
| --- | ---: | ---: | ---: | ---: |
| `__init__.py` | 3/9 | 3/9 | 0/2 | 0/2 |
| `adapters/__init__.py` | 10/10 | 10/10 | 2/2 | 2/2 |
| `adapters/langgraph.py` | 68/88 | 68/88 | 25/42 | 25/42 |
| `adapters/llamaindex.py` | 61/82 | 61/82 | 22/38 | 22/38 |
| `adapters/openai_agents.py` | 46/146 | 46/146 | 18/80 | 18/80 |
| `assertions/__init__.py` | 2/2 | 2/2 | 0/0 | 0/0 |
| `assertions/engine.py` | 151/185 | 153/185 | 54/82 | 54/82 |
| `cli.py` | 26/68 | 26/68 | 3/14 | 3/14 |
| `config.py` | 134/148 | 134/148 | 32/42 | 33/42 |
| `plugin.py` | 32/87 | 86/87 | 0/24 | 24/24 |
| `recorder/__init__.py` | 2/2 | 2/2 | 0/0 | 0/0 |
| `recorder/core.py` | 72/78 | 72/78 | 14/16 | 14/16 |
| `recorder/interceptors.py` | 125/135 | 125/135 | 39/48 | 39/48 |
| `replay/__init__.py` | 2/2 | 2/2 | 0/0 | 0/0 |
| `replay/engine.py` | 89/107 | 95/107 | 30/42 | 34/42 |
| `serialization.py` | 116/131 | 116/131 | 47/52 | 47/52 |
| `types.py` | 67/71 | 67/71 | 0/0 | 0/0 |

## Plugin behavior now asserted

`tests/unit/test_plugin.py` enables pytester locally. Each temporary pytest
project disables entry-point autoload and explicitly registers
`agentcontract.plugin` once, with `--strict-markers`. In-process execution keeps
fixture calls visible to the outer coverage collector. Files stay in pytester's
temporary directories; recording uses manually supplied turns, without network.

- Live, record, replay, and record precedence in `ac_mode` when both flags are set.
- Explicit configuration takes precedence over discovered configuration.
- Both markers register; positional names, keyword names, alias forms, and the
  unmarked test-name fallback resolve for both recording and replay.
- Default and CLI-overridden directories produce loadable cassettes preserving
  turns and summary counts. Replay consumes the recorded tool result and does
  not overwrite the cassette; live/replay recorders do not save new files.
- Inactive replay returns `None`; missing files skip before the test body;
  malformed JSON produces a setup error with path and exception type.
- Injected `OSError`, `ValueError`, and `TypeError` at save/load boundaries expose
  diagnostics. Save failures are teardown errors after a passing body; load
  failures are setup errors. No permission-dependent filesystem tricks are used.
- Contract checks combine defaults, matching scenario assertions, extras, and
  policies. Assertions verify individual default/scenario/extra/policy failures,
  passing results, unmatched scenarios, repeated calls, and unchanged config.

The sole remaining unexecuted plugin statement is line 138, `return None` after
`pytest.skip()`. Skip raises, so this return is unreachable in normal pytest
execution. It is not a missing branch and is not artificially covered.

## Prioritized remaining gaps

Ranked by user impact, missing branch destinations, and the strength of existing
assertions, rather than by percentage alone:

1. **Adapter interception lifecycle:** `adapters/openai_agents.py` has 62/80
   branch destinations missing; lines 38–91 (SDK import, sync/async patching,
   restoration) are entirely unexecuted. Existing tests exercise selected result
   extraction shapes, not the patch lifecycle. Add fake-SDK tests for invocation,
   exceptions, restoration, missing/noncallable methods, usage and tool-output
   shapes. LangGraph and LlamaIndex have 17 and 16 missing destinations,
   respectively, including async paths and defensive extraction. Existing sync
   mock tests do not establish compatibility with real SDK versions.
2. **Assertion correctness:** `assertions/engine.py` has 28/82 missing branch
   destinations. Target resolution for full conversations, indexed turns, tool
   arguments/results, malformed targets, and absent assistant output remains
   substantially unexecuted (lines 113–161). Existing happy-path and selected
   failure tests do not prove every target/type combination. Add a table of
   resolved values and explicit failure results, plus invalid regex/schema and
   non-integer count inputs. Incorrect resolution could yield misleading passes.
3. **CLI usability and failure handling:** `cli.py` has 11/14 missing destinations.
   Its sole dedicated test asserts invalid-cassette `info` failure. Successful
   `info`, `validate`, `init`, help, missing paths, refusal to overwrite config,
   write failures, and process exit codes remain largely unexecuted. Validate
   generated config semantically and assert stdout/stderr and exit status.
4. **Replay mismatch diagnostics:** `replay/engine.py` has 8/42 missing destinations,
   including unused stubs, extra turns without tools, role/function mismatches,
   and missing turns without tools. Existing tests cover matching, argument
   mismatches, exhaustion and tool counts; add exact counter/error expectations
   for these alternatives and unprintable/long content previews.
5. **Configuration and serialization boundaries:** config has 9/42 missing
   destinations, including fallback discovery and numeric coercion edge cases;
   serialization has 5/52, including optional fields and malformed nested data.
   Existing null/coercion/round-trip tests are useful but not exhaustive. Add
   invalid/overflow/bool numeric cases and meaningful round-trip invariants.
6. **Recorder and SDK interceptors:** core has 2/16 missing destinations
   (optional numeric handling and suffix creation); interceptors have 9/48,
   mainly missing SDK members and malformed response shapes. Add errors and
   restoration checks; executed lines alone do not prove cleanup after failures.
7. **Public API and data helpers:** top-level lazy imports have 2/2 destinations
   missing; `types.py` has four unexecuted helper lines. Test advertised imports,
   unknown-name errors, and helper output. Package initializers with 100%
   execution have no implication of complete public API behavior coverage.

## Limitations and suspected inconsistencies

- Both CLI flags yield `ac_mode == "record"`, but recorder and replay fixtures
  independently inspect their own flags. Replay can therefore still load/skip
  while mode says record. Only the documented current mode precedence is asserted;
  defining a single precedence rule across fixtures is a separate production task.
- `--ac-scenarios` help says it overrides config, but recorder/replay fixtures
  use the CLI option or hardcoded `tests/scenarios`, without consulting config's
  scenario include patterns. Configuration-derived cassette placement needs a
  product decision and separate regression test.
- Replay's docstring promises `None` for missing files; actual behavior is a skip,
  now asserted. Documentation correction is outside this bounded change.
- Marker conflicts, malformed/empty marker arguments, parent marker precedence,
  failing test bodies during recording, invalid explicit config, and installed
  wheel/entry-point packaging in a separate process remain behavioral gaps even
  though all plugin branch destinations execute. No live provider compatibility,
  network integration, or full Python/OS matrix was tested.

## Reproduction and checks

From an isolated checkout of the target revision, create a fresh environment
(the audit used `/tmp/ac-test-gap-env`) and install only `.[dev]`:

```sh
uv venv /tmp/ac-test-gap-env --python 3.12
uv pip install --python /tmp/ac-test-gap-env/bin/python -e '.[dev]'
/tmp/ac-test-gap-env/bin/pytest tests/unit/test_plugin.py -q
/tmp/ac-test-gap-env/bin/ruff check src/ tests/
/tmp/ac-test-gap-env/bin/mypy src/agentcontract/ --ignore-missing-imports
```

Then run either coverage command above with unique temporary output filenames.
For an exact baseline, use the base SHA in another worktree and install that
worktree into its own environment. Do not reuse an editable install pointing at
another checkout. Future dependency resolution may differ from the versions
recorded above.

CI lint passes. CI's advisory mypy command reports the same **nine pre-existing
errors in four files** before and after: missing return annotation in
`__init__.py`, `Any` return in `serialization.py`, four optional-member errors in
`recorder/interceptors.py`, and three unused ignores in `adapters/openai_agents.py`.
The complete diagnostic outputs were compared and are identical. CI marks this
step `continue-on-error`; this PR does not change that policy or fix those errors.
