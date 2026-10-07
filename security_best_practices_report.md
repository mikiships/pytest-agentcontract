# Security anti-pattern review

Reviewed revision: `17c52033d3cadefe7c1151b6c76c74995b0db71a` (`origin/main`). Review date: 2026-10-07.

## Executive summary

Four actionable, medium-severity security foot-guns were confirmed with synthetic inputs: replay mode does not prevent live client calls; confirmation checks accept refusal; malformed policy sections silently disable checks; and sensitive trajectory data is persisted and included in diagnostics without redaction. These are conditional developer/CI safety failures, not demonstrated remote exploitation. No critical or high-severity issue was confirmed. This report proposes follow-up fixes only; it changes no implementation or dependencies.

## Scope and trust assumptions

Reviewed configuration discovery/parsing, JSON serialization, CLI, pytest fixtures, recorder and SDK interceptors, replay engine, all three framework adapters, assertions/policies, relevant unit tests, README, dependency declarations, and both CI workflows. Source locations below refer to the reviewed revision.

Python tests, installed packages, SDK objects, configuration, explicit output paths, and pytest marker definitions are trusted local inputs. A contributor able to replace executable tests already has code execution; this report does not label that access a new vulnerability. Prompts, model responses, tool arguments/results, and imported cassettes can contain sensitive or externally supplied data. Sharing cassettes or CI diagnostics crosses a confidentiality boundary. Policies are retrospective assertions, not runtime authorization controls. Replay stubs return data and do not execute recorded function names.

The security-best-practices skill has Python web-framework guidance but no reference specific to this non-web pytest plugin. Findings therefore rely on repository evidence and bounded reproductions, not an assumed web threat model.

## Methodology

Searched for unsafe YAML/object loading, eval/exec, shell/subprocess calls, input-derived imports, embedded credential patterns, disabled TLS verification, temporary-file handling, and file writes. Followed candidate data flows through callers and compared tests with documented behavior. Used fake clients and pytest temporary directories; no real credentials, external model calls, destructive payloads, or resource-exhaustion workloads were used. The reproduction appendix is embedded documentation, not an added test suite.

## Confirmed findings (medium severity, in recommended follow-up order)

### SF-001 — Replay flag does not enforce the advertised offline boundary

**Locations:** `src/agentcontract/plugin.py:28–32,67–74,94–104,111–146`; `src/agentcontract/recorder/interceptors.py:32–40,72–80`; `src/agentcontract/config.py:134`; README “SDK Auto-Recording” example.

**Trigger:** Run a test that uses the documented recording interceptor under `--ac-replay` without explicitly branching around the original SDK call. The recorder fixture remains available, and the interceptor unconditionally calls `original_create`. `stub_tools` is parsed but is not used to install or enforce stubs. Supplying both flags also selects record mode first.

**Impact:** A developer relying on the CLI's “no network” guarantee can transmit prompts, spend API credits, or execute application tool side effects when credentials/services are available. With no credentials such tests may simply fail. The replay engine itself has no automatic live fallback; the risk is the surrounding integration and misleading guarantee.

**Safe reproduction:** `test_replay_still_calls_original` below uses a fake client counter under actual pytest replay mode. The counter increments once. No network is contacted.

**Recommendation:** Make record/replay flags mutually exclusive; fail clearly when recording interceptors are used in replay without a replay integration. Provide an explicit replay client/tool integration that fails on unstubbed calls, and an opt-in network-denial mechanism with documented coverage. Correct README examples to show supported manual stub wiring (the current `ac_replay_engine.run()` example refers to a nonexistent method). Clarify that arbitrary application tools require separate side-effect isolation.

**Existing protection:** `ToolStub.get_result` raises on exhaustion or argument mismatch; missing fixture cassettes skip the test. Those protections do not intercept unrelated client/tool calls.

### SF-002 — Explicit refusal passes `requires_confirmation`

**Location:** `src/agentcontract/assertions/engine.py:355–383`.

**Trigger:** A protected tool follows any user turn, including “No, do not refund this order.” The checker tests the previous role only, without checking approval or binding approval to the operation/arguments.

**Impact:** A contract intended to detect unauthorized operations reports success for an agent that acts after refusal. This is a false-negative safety check; the plugin does not itself execute or authorize the operation.

**Safe reproduction:** `test_refusal_counts_as_confirmation` below produces a passing result for that exact refusal followed by a recorded `process_refund` call.

**Recommendation:** Define structured confirmation evidence bound to the requested operation and arguments and require it for this policy. If only adjacency is intended, rename the policy and document its limited meaning. Add regression coverage for refusal, unrelated user messages, and approval of a different operation. Do not substitute a loose keyword match for authorization.

**Existing protection:** `TestRequiresConfirmationPolicy` covers preceding-user success and preceding-assistant failure; it does not establish affirmative consent.

### SF-003 — Malformed policy sections silently remove security checks

**Locations:** `src/agentcontract/config.py:102–123,190–202`; `src/agentcontract/assertions/engine.py:33–34,62–68`; `src/agentcontract/plugin.py:167–184`.

**Trigger:** A trusted developer accidentally writes `policies` as a mapping instead of a list (for example, omits a YAML list dash). `_coerce_list` replaces it with `[]`; non-dictionary list entries are also dropped. Contract evaluation then passes with no results if no other checks fail. Default assertion lists have analogous coercion.

**Impact:** CI can pass an agent's disallowed tool use despite an apparently configured allowlist. This is a configuration-error foot-gun, not a privilege escalation for an attacker who already controls the tests/configuration.

**Safe reproduction:** `test_malformed_policy_section_disappears` below checks a prohibited refund against malformed YAML: policies become empty and the contract passes. Correcting the same policy to a list rejects the call.

**Recommendation:** Validate present policy/assertion sections strictly and raise a configuration error for wrong container types or invalid entries. Distinguish omitted optional sections from malformed supplied sections; optionally require at least one evaluated check for a configured contract. Preserve fail-closed behavior for unknown types.

**Existing protection:** Unknown policy types and exceptions fail closed once a policy reaches the engine; `test_config_from_dict_coerces_null_policy_lists` also verifies an empty *tool allowlist* rejects calls. Neither covers a missing effective policy caused by section coercion.

### SF-004 — Sensitive trajectory values reach cassettes and failure diagnostics verbatim

**Locations:** `src/agentcontract/recorder/core.py:118–145,178–194`; `src/agentcontract/serialization.py:176–184,304–308`; `src/agentcontract/plugin.py:99–104`; `src/agentcontract/replay/engine.py:69–73`; `src/agentcontract/assertions/engine.py:182`.

**Trigger:** A caller records sensitive content, arguments, or results and saves/shares the cassette, or exposes a mismatch exception in CI output. The save path has no redaction step. A stub mismatch interpolates both full argument dictionaries; exact-match assertion failures also interpolate actual content.

**Impact:** Private prompts, tokens contained in payloads, customer data, or tool outputs can enter version control, artifacts, or retained logs. Exposure depends on sensitive inputs and subsequent sharing; no actual leaked credential was discovered. OpenAI/Anthropic interceptors record response content/tool requests, not SDK authentication headers or every input prompt. Manual recording and the LangGraph adapter can record user messages; LlamaIndex can capture tool outputs.

**Safe reproduction:** `test_sensitive_persistence_and_log` below saves three synthetic markers and verifies each survives in JSON. A mismatch then exposes both recorded and current synthetic argument markers in the exception text.

**Recommendation:** Add a configurable sanitization hook before persistence, recursively handle nested tool payloads and free-form content, and provide safe diagnostic formatting that omits values by default. Document synthetic-fixture use and cassette review before committing; key-name redaction alone cannot sanitize arbitrary prose. Keep replay matching semantics explicit when sanitizing values. Consider restrictive file creation permissions for sensitive local recordings.

**Existing protection:** Round-trip and adapter tests establish faithful capture rather than confidentiality. No automatic redaction was found in the reviewed path.

## Hardening opportunities and excluded false positives

These are not additional confirmed vulnerabilities under the stated trust model.

- **Output containment:** `plugin.py:102,134` joins a local scenario name directly with the scenario directory; `serialization.py:304–308` creates directories and truncates the supplied destination. The bounded path test below demonstrates `../outside` escaping the nominal scenario folder, but only within its temporary directory. No external-input-to-marker path was established. Reject absolute/traversal scenario names and use a resolved containment check for fixture-generated paths. Explicit `Recorder.save(path)` is intentionally caller-directed. Symlink/race exploitation was not tested; consider atomic, no-clobber writes where appropriate. CLI init checks existence but is not race-proof (`cli.py:88–92,125`).
- **Parsing/validation:** YAML uses `safe_load` (`config.py:85`), JSON uses `json.load` (`serialization.py:314`); a harmless Python-object YAML tag was rejected. No pickle, eval/exec, shell execution, disabled TLS verification, or unsafe temporary-file API was found in production source. Lazy imports select from constant allowlists. JSON normalization can accept incomplete data, and CLI `validate` is a loadability check rather than complete schema validation. Consider strict cassette validation and input size/depth limits if accepting untrusted artifacts; no exhaustion test was run.
- **Assertion resource/network boundaries:** Regex patterns and JSON Schemas are trusted configuration. Regex worst-case runtime and external schema-reference resolution deserve explicit limits if configuration becomes untrusted. No remote-reference or denial-of-service vulnerability was validated.
- **CI/dependencies:** CI runs tests/lint on pull requests; publishing uses release-triggered trusted publishing. Dependencies have lower bounds and actions use version tags. Dependency CVE auditing, action supply-chain verification, and hosted repository permissions were outside this bounded review. No claim of current dependency vulnerability or overall security certification is made.

## Validation and limitations

On Python 3.14.7, using an existing environment without installing or changing dependencies, all **77 existing tests passed**. All **6 bounded reproduction/control checks passed**, then passed again when extracted from this report. `PYTHONPATH=src` selected the isolated reviewed checkout; plugin autoload was disabled and `agentcontract.plugin` explicitly loaded. The CI Python 3.10–3.13 matrix was not reproduced. No live SDK/framework integration or actual unauthorized operation was exercised. The report is evidence of observed behavior at the stated revision, not a comprehensive audit.

Commands (use a Python environment containing the project's declared test dependencies):

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p agentcontract.plugin tests/ -q
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p agentcontract.plugin /tmp/security_footgun_repro.py --ac-replay -q
```

Copy the following appendix to `/tmp/security_footgun_repro.py`. All values are synthetic and writes use pytest-managed temporary directories.

## Reproduction appendix

```python
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from agentcontract.config import AgentContractConfig, PolicySpec
from agentcontract.assertions.engine import AssertionEngine
from agentcontract.recorder.core import Recorder
from agentcontract.recorder.interceptors import patch_openai
from agentcontract.replay.engine import ReplayEngine, ToolStubArgumentsMismatch, ToolStubExhausted
from agentcontract.serialization import load_run


def test_sensitive_persistence_and_log(tmp_path):
    r = Recorder()
    r.add_turn('user', content='SYNTHETIC_PRIVATE_PROMPT')
    r.add_turn('assistant', tool_calls=[{'id':'1', 'function':'lookup', 'arguments':{'token':'SYNTHETIC_ARGUMENT'}, 'result':'SYNTHETIC_RESULT'}])
    p = r.save(tmp_path / 'sample.agentrun.json')
    assert all(s in p.read_text() for s in ['SYNTHETIC_PRIVATE_PROMPT','SYNTHETIC_ARGUMENT','SYNTHETIC_RESULT'])
    with pytest.raises(ToolStubArgumentsMismatch) as e:
        ReplayEngine(load_run(p)).tool_stub.get_result('lookup', {'token':'SYNTHETIC_ACTUAL'})
    assert 'SYNTHETIC_ARGUMENT' in str(e.value) and 'SYNTHETIC_ACTUAL' in str(e.value)


def test_refusal_counts_as_confirmation():
    r = Recorder()
    r.add_turn('user', content='No, do not refund this order.')
    r.add_turn('assistant', tool_calls=[{'function':'process_refund','arguments':{}}])
    assert AssertionEngine().check(r.run, policies=[PolicySpec(name='confirm',type='requires_confirmation',tools=['process_refund'])]).passed


def test_malformed_policy_section_disappears():
    cfg = AgentContractConfig.from_dict(yaml.safe_load('policies:\n  name: allowed\n  type: tool_allowlist\n  tools: [lookup]\n'))
    r = Recorder()
    r.add_turn('assistant', tool_calls=[{'function':'process_refund'}])
    assert cfg.policies == []
    assert AssertionEngine().check(r.run, policies=cfg.policies).passed
    valid = AgentContractConfig.from_dict({'policies':[{'name':'allowed','type':'tool_allowlist','tools':['lookup']}]})
    assert not AssertionEngine().check(r.run, policies=valid.policies).passed


def test_replay_still_calls_original(ac_mode, ac_recorder):
    assert ac_mode == 'replay'
    calls = []
    def original(**kwargs):
        calls.append('called')
        return {'choices':[{'message':{'content':'fake'}}]}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=original)))
    undo = patch_openai(client, ac_recorder)
    try:
        client.chat.completions.create()
    finally:
        undo()
    assert calls == ['called']


def test_safety_controls(tmp_path):
    p = tmp_path / 'config.yml'
    p.write_text('!!python/object:builtins.object {}')
    with pytest.raises(yaml.constructor.ConstructorError):
        AgentContractConfig.from_file(p)
    with pytest.raises(ToolStubExhausted):
        ReplayEngine(Recorder().run).tool_stub.get_result('missing')


def test_path_containment_hardening(tmp_path):
    base = tmp_path / 'scenarios'
    base.mkdir()
    scenario = '../outside'
    Recorder().save(base / f'{scenario}.agentrun.json')
    assert (tmp_path / 'outside.agentrun.json').exists()
```
