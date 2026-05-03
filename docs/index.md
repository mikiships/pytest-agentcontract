# pytest-agentcontract Documentation

pytest-agentcontract records LLM agent trajectories into `.agentrun.json`
cassettes, replays them without live model or tool calls, and checks the run
against assertions and policies.

The workflow is:

1. Record a scenario with `pytest --ac-record`.
2. Commit the generated cassette.
3. Replay the scenario with `pytest --ac-replay`.
4. Assert tool usage, response content, and safety policies with
   `ac_check_contract`.

## Guides

- [Quickstart](quickstart.md): installation, pytest options, markers, fixtures,
  and a compact customer-support example.
- [Configuration](configuration.md): supported `agentcontract.yml` sections and
  default values.
- [Assertions and Policies](assertions-and-policies.md): assertion types, target
  syntax, policy types, and practical examples.
- [Replay and Cassettes](replay-and-cassettes.md): cassette structure, recorder
  output, replay tool stubs, validation, and CI usage.
- [Integrations](integrations.md): manual recording, OpenAI and Anthropic SDK
  interceptors, and framework adapter entry points.

## Command Reference

```bash
pytest --ac-record
pytest --ac-replay
pytest --ac-config path/to/agentcontract.yml
pytest --ac-scenarios tests/scenarios

agentcontract init
agentcontract info tests/scenarios/refund-eligible.agentrun.json
agentcontract validate tests/scenarios/refund-eligible.agentrun.json
```
