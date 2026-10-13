# Test coverage gaps

Measured on Python 3.12.14, pytest 9.0.3 and coverage.py 7.14.0 against
default branch main at 17c5203. The original checkout remains on
nightshift/lint-fix-ruff-2 with its pre-existing uv.lock modification;
work uses the isolated nightshift/test-gap-plugin branch.

## Results

Baseline: 77 tests passed. After: 102 tests passed (25 new integration cases).
No baseline test failures. No production code changes.
Percentages below include statements and branches, not statements alone.

| Scope | Before | After | Statements covered after | Branches covered after |
| --- | ---: | ---: | ---: | ---: |
| Repository | 70.41% | 75.53% | 1069/1351 | 317/484 |
| Plugin | 28.83% | 99.10% | 86/87 | 24/24 |

Baseline plugin execution covered imports, fixture declarations, option and marker
registration during pytest startup; it did not assert public fixture behavior.
The integration suite now asserts:

- CLI help, strict acceptance of both markers, live/record/replay modes, and
  record precedence when both flags are supplied.
- Explicit configuration selection over discovered configuration.
- Both marker names with positional and keyword scenario names, plus test-name
  fallback, for recording and replay in default and overridden directories.
- Metadata/filename agreement, synthetic turns, finalized summaries and token
  totals, automatic persistence only in record mode, and loadable cassettes.
- Save failures for OSError, ValueError and TypeError as teardown errors with
  cassette path and underlying exception.
- Disabled replay, usable replay tool stubs, missing-cassette skips, malformed JSON
  and unreadable cassette setup errors.
- Default assertions, matching/unmatched scenario overrides, extra assertions,
  policy failures, repeated-call isolation and unchanged configuration lists.

Nested pytest sessions run in-process under the outer coverage collector with
plugin autoload disabled and exactly one explicit agentcontract plugin load.
Recording and replay use distinct test module files to prevent import-cache reuse.
Coverage confirms execution of successful replay construction (plugin.py:146),
both marker branches (127,129), teardown save/error paths (104-106), and contract
merging (169-187). All 24 plugin branches are covered. The only missing plugin
statement is line 138, an unreachable return after pytest.skip raises.

## Prioritized follow-up inventory

Coverage indicates execution, not correctness; these are candidate tests requiring
behavioral assertions, not a target to mechanically reach 100%.

| Priority | Location | Uncovered behavior / user impact | Proposed regression |
| --- | --- | --- | --- |
| P1 | adapters/openai_agents.py:38-91,189-270 | Runner patch/unpatch and SDK response variants can lose trajectories | Fake SDK module; synchronous/asynchronous runner calls, exception cleanup, nested messages and malformed tool arguments |
| P1 | cli.py:33-130 | CLI dispatch, info/validate failures and init may misreport or overwrite user files | Temporary valid/invalid cassettes; assert exit status, output, generated config and existing-file preservation |
| P1 | assertions/engine.py:87-159,188-216,251-290,364 | Unknown assertions, target resolution, malformed specs and confirmation failures can undermine contracts | Parameterized invalid targets/specs, missing values, tool result targets and intervening non-user confirmation turns |
| P2 | adapters/langgraph.py:40-78,95-159; adapters/llamaindex.py:40-175 | Alternate message shapes and patch lifecycle can silently omit turns/tools | Fake graph/agent objects, async failures, unpatch restoration, null/list content and tool variants |
| P2 | replay/engine.py:14-20,135-188 | Preview errors, unused stubs and turn/tool mismatches need exact diagnostic assertions | Unprintable/long content, incomplete consumption, wrong role/name/arguments and missing calls |
| P2 | serialization.py:68-116,170-207,291-292 | Coercion and filesystem failure paths affect cassette portability | Invalid scalar fields, absent/invalid dates, write failures and schema validation errors |
| P2 | recorder/interceptors.py:29-217 | Optional SDK capability branches and malformed responses affect recording | Fake SDKs with missing methods and malformed usage/content, verify restoration after failures |
| P3 | config.py:163,218-265 | Discovery fallback and numeric/boolean edge cases may choose unexpected defaults | No-config ancestor walk; booleans, fractional/infinite values and invalid scalar conversions |
| P3 | recorder/core.py:34-45,187 | Invalid numeric inputs and extensionless save paths | Synthetic malformed latency/token values and extensionless roundtrip |
| P3 | __init__.py:11-22; types.py:108-117 | Lazy public exports and derived helpers lack direct assertions | Import every public lazy export, reject unknown attributes and check helper outputs |
| P3 | plugin.py:138 and additional semantics | Unreachable statement; coverage does not prove marker precedence or failed-body persistence | Leave unreachable code unchanged; follow up with conflicting/inherited markers and record teardown after test-body failure |

## Reproduction and validation

Use a fresh environment with editable install of this worktree. Store artifacts
outside the repository; do not reuse a pre-existing .coverage file. Run baseline
at 17c5203, then the same commands on this branch with a different data filename:

```sh
uv venv /tmp/test-gap-venv
uv pip install --python /tmp/test-gap-venv/bin/python -e '.[dev]'
mkdir -p /tmp/test-gap-coverage/reports
export COVERAGE_FILE=/tmp/test-gap-coverage/after
/tmp/test-gap-venv/bin/python -m coverage run --branch --source=src/agentcontract -m pytest tests -q
/tmp/test-gap-venv/bin/python -m coverage report -m
/tmp/test-gap-venv/bin/python -m coverage json -o /tmp/test-gap-coverage/reports/after.json
/tmp/test-gap-venv/bin/python -m pytest tests/integration -q
/tmp/test-gap-venv/bin/ruff check src/ tests/
/tmp/test-gap-venv/bin/mypy src/agentcontract/ --ignore-missing-imports
```

Focused suite: 25 passed. Full suite: 102 passed. Repository-wide Ruff passes.
CI's non-blocking mypy check reports nine existing errors in unchanged source:
missing return annotation (__init__.py:9), Any return (serialization.py:129),
optional attribute access (recorder/interceptors.py:43,46,77,80), and unused
ignores (adapters/openai_agents.py:42,65,82). These are separate from test gaps.
Local validation covers Python 3.12; CI's Python 3.10–3.13 matrix remains authoritative.

## Repository-wide measured missing lines and branches

All source modules are included, including those without gaps. Ranges below use
source line numbers on main at 17c5203. Branch pairs are source → destination;
negative destinations denote function exits.

| Module | Before | After | Remaining missing lines | Remaining missing branches |
| --- | ---: | ---: | --- | --- |
| __init__.py | 27.3% | 27.3% | 11, 17, 18, 20, 21, 22 | 17→18, 17→22 |
| adapters/__init__.py | 100.0% | 100.0% | — | — |
| adapters/langgraph.py | 71.5% | 71.5% | 40, 45, 47, 49, 67, 68, 69, 70, 71, 95, 122, 130, 134, 135, 136, 137, 138, 151, 152, 153 | 39→40, 44→45, 46→47, 48→49, 51→63, 63→75, 76→78, 78→-75, 94→95, 102→97, 118→122, 129→130, 131→134, 134→135, 134→138, 150→151, 159→149 |
| adapters/llamaindex.py | 69.2% | 69.2% | 40, 48, 50, 62, 63, 64, 65, 66, 87, 108, 109, 110, 134, 168, 169, 170, 171, 172, 173, 174, 175 | 39→40, 47→48, 49→50, 86→87, 106→108, 108→109, 108→113, 115→-96, 133→134, 153→156, 168→169, 168→170, 171→172, 171→173, 173→174, 173→175 |
| adapters/openai_agents.py | 28.3% | 28.3% | 38, 39, 41, 42, 43, 44, 48, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 65, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 82, 84, 85, 87, 88, 89, 91, 108, 109, 144, 145, 146, 147, 148, 149, 179, 184, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 202, 208, 214, 215, 216, 217, 218, 224, 230, 231, 232, 233, 234, 235, 237, 238, 239, 240, 241, 242, 245, 246, 247, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 260, 265, 266, 267, 268, 269, 270 | 38→39, 38→41, 52→53, 52→54, 54→55, 54→68, 69→70, 69→71, 71→72, 71→84, 84→85, 84→87, 88→-87, 88→89, 107→108, 113→-94, 134→126, 142→144, 145→126, 145→146, 148→126, 148→149, 166→126, 178→179, 183→184, 186→189, 189→190, 189→202, 191→192, 191→199, 192→193, 192→196, 193→191, 193→194, 197→191, 197→198, 207→208, 211→214, 215→216, 215→224, 217→215, 217→218, 230→231, 230→245, 232→233, 232→234, 234→230, 234→235, 239→230, 239→240, 246→247, 246→260, 248→249, 248→250, 250→251, 250→260, 255→256, 255→260, 266→267, 266→270, 267→268, 267→269 |
| assertions/__init__.py | 100.0% | 100.0% | — | — |
| assertions/engine.py | 76.8% | 77.9% | 87, 95, 96, 113, 114, 115, 124, 127, 128, 129, 130, 131, 134, 135, 136, 137, 138, 139, 140, 141, 146, 155, 156, 157, 159, 188, 203, 216, 251, 284, 290, 364 | 86→87, 112→113, 114→115, 114→117, 118→124, 126→127, 128→129, 128→131, 129→128, 129→130, 133→134, 139→140, 139→141, 143→159, 145→146, 150→157, 153→155, 155→151, 155→156, 187→188, 202→203, 215→216, 250→251, 264→260, 283→284, 289→290, 363→364 |
| cli.py | 35.4% | 35.4% | 33, 34, 35, 36, 38, 39, 47, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 72, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 89, 90, 91, 92, 94, 124, 125, 126, 127, 128, 129, 130, 134 | 31→33, 33→34, 33→35, 35→36, 35→38, 46→47, 74→75, 74→78, 90→91, 90→94, 133→134 |
| config.py | 87.4% | 87.9% | 163, 218, 222, 233, 235, 238, 239, 247, 249, 252, 253, 261, 264, 265 | 159→163, 217→218, 221→222, 223→225, 232→233, 234→235, 246→247, 248→249, 260→261 |
| plugin.py | 28.8% | 99.1% | 138 | — |
| recorder/__init__.py | 100.0% | 100.0% | — | — |
| recorder/core.py | 91.5% | 91.5% | 34, 35, 41, 44, 45, 187 | 40→41, 186→187 |
| recorder/interceptors.py | 89.6% | 89.6% | 29, 63, 110, 114, 178, 188, 208, 214, 216, 217 | 28→29, 62→63, 109→110, 113→114, 174→169, 177→178, 187→188, 207→208, 213→214 |
| replay/__init__.py | 100.0% | 100.0% | — | — |
| replay/engine.py | 79.9% | 86.6% | 14, 17, 18, 20, 135, 136, 148, 149, 160, 170, 187, 188 | 13→14, 19→20, 134→135, 147→148, 159→160, 167→170, 186→187, 199→202 |
| serialization.py | 89.1% | 90.2% | 68, 69, 78, 79, 88, 89, 113, 114, 116, 170, 191, 207, 291, 292 | 110→116, 169→170, 190→191, 206→207 |
| types.py | 94.4% | 94.4% | 108, 110, 115, 117 | — |
