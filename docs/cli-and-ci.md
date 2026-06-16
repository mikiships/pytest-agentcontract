# CLI And CI

pytest-agentcontract includes a small `agentcontract` CLI and pytest options for record/replay workflows.

## CLI Commands

Create a starter config in the current directory:

```bash
agentcontract init
```

`init` writes `agentcontract.yml` and exits with an error if the file already exists.

Show a cassette summary:

```bash
agentcontract info tests/scenarios/refund-eligible.agentrun.json
```

`info` prints scenario, run id, recorded timestamp, model, turn count, tool-call count, duration, tokens, and estimated cost.

Validate that a cassette can be loaded:

```bash
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
```

`validate` is a structural loader check. It does not run configured assertions or policies.

## Local Pytest Commands

Record or update one scenario:

```bash
pytest tests/test_refund.py --ac-record -k refund_eligible
```

Replay all committed cassettes:

```bash
pytest --ac-replay
```

Use a non-default scenario directory:

```bash
pytest --ac-record --ac-scenarios examples/customer_support/scenarios
pytest --ac-replay --ac-scenarios examples/customer_support/scenarios
```

Use a specific config file:

```bash
pytest --ac-replay --ac-config agentcontract.yml
```

The example project can be replayed with:

```bash
pytest examples/customer_support --ac-replay --ac-scenarios examples/customer_support/scenarios
```

## Cassette Artifact Handling

Commit stable cassette files that define expected behavior:

```text
tests/scenarios/*.agentrun.json
```

Treat cassette changes as behavior changes. Review tool names, argument objects, tool results, final responses, and token/timing changes before merging.

Use CI artifacts for generated reports or temporary diagnostics, not as the source of truth for replay. The default parsed reporting artifact path is `agentci-results/`.

Do not record new cassettes in normal CI runs. CI should use `--ac-replay` against cassettes already committed to the branch.

## Minimal GitHub Actions Example

```yaml
name: agent contracts

on:
  pull_request:
  push:
    branches: [main]

jobs:
  replay:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install package
        run: python -m pip install -e ".[all]"

      - name: Validate cassettes
        run: |
          for cassette in tests/scenarios/*.agentrun.json; do
            agentcontract validate "$cassette"
          done

      - name: Replay contracts
        run: python -m pytest --ac-replay

      - name: Upload diagnostics
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: agentci-results
          path: agentci-results/
          if-no-files-found: ignore
```

If your CI only needs the base package, install `python -m pip install -e .` instead of `.[all]`.
