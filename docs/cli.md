# CLI

pytest-agentcontract provides a small `agentcontract` CLI and pytest plugin
options.

## `agentcontract init`

Create a starter `agentcontract.yml` in the current directory:

```bash
agentcontract init
```

The command exits with an error if `agentcontract.yml` already exists.

The generated file includes scenario globs, replay defaults, one placeholder
assertion, an allowlist policy, budgets, and reporting defaults.

## `agentcontract info`

Print a summary for a cassette:

```bash
agentcontract info tests/scenarios/refund-eligible.agentrun.json
```

Output includes scenario, run id, recording timestamp, model, turn count, tool
call count, duration, tokens, and estimated cost. The command exits nonzero when
the file is missing or cannot be loaded.

## `agentcontract validate`

Check that a cassette can be loaded:

```bash
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
```

A valid cassette prints the scenario name and turn count. An invalid cassette
prints the load error and exits nonzero.

## Pytest Options

| Option | Behavior |
| --- | --- |
| `--ac-record` | Sets `ac_mode` to `"record"` and auto-saves `ac_recorder` after each test. |
| `--ac-replay` | Sets `ac_mode` to `"replay"` and loads the matching cassette into `ac_replay_engine`. |
| `--ac-config PATH` | Loads a specific `agentcontract.yml` instead of discovering one. |
| `--ac-scenarios DIR` | Uses `DIR` for cassette reads and writes instead of `tests/scenarios`. |

If neither `--ac-record` nor `--ac-replay` is set, `ac_mode` is `"live"`.

## Common Commands

```bash
# Record all marked tests.
pytest --ac-record

# Replay all marked tests from tests/scenarios.
pytest --ac-replay

# Record to a custom scenario directory.
pytest --ac-record --ac-scenarios examples/customer_support/scenarios

# Replay with a custom config file.
pytest --ac-replay --ac-config agentcontract.yml
```

`--ac-record` and `--ac-replay` are independent boolean options. Use one mode at
a time so test code has an unambiguous `ac_mode` path.
