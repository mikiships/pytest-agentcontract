"""Tests for config parsing."""

from pathlib import Path

from agentcontract.config import AgentContractConfig


def test_config_from_dict():
    raw = {
        "version": "1",
        "scenarios": {"include": ["tests/**/*.agentrun.json"]},
        "replay": {"stub_tools": True, "concurrency": 3},
        "defaults": {
            "assertions": [
                {"type": "contains", "target": "final_response", "value": "refund"}
            ]
        },
        "overrides": {
            "denied-flow": {
                "assertions": [
                    {"type": "not_called", "target": "tool:process_refund"}
                ]
            }
        },
        "policies": [
            {"name": "tools", "type": "tool_allowlist", "tools": ["lookup_order"]}
        ],
    }

    config = AgentContractConfig.from_dict(raw)
    assert config.version == "1"
    assert config.scenario_include == ["tests/**/*.agentrun.json"]
    assert config.replay.concurrency == 3
    assert len(config.default_assertions) == 1
    assert config.default_assertions[0].type == "contains"
    assert "denied-flow" in config.overrides
    assert len(config.policies) == 1
    assert config.policies[0].tools == ["lookup_order"]


def test_config_from_yaml(tmp_path: Path):
    yaml_content = """\
version: "1"
scenarios:
  include: ["tests/scenarios/**/*.agentrun.json"]
replay:
  stub_tools: true
defaults:
  assertions:
    - type: contains
      target: final_response
      value: "hello"
"""
    config_file = tmp_path / "agentcontract.yml"
    config_file.write_text(yaml_content)

    config = AgentContractConfig.from_file(config_file)
    assert config.replay.stub_tools is True
    assert config.default_assertions[0].value == "hello"


def test_config_defaults():
    config = AgentContractConfig()
    assert config.suite_pass_rate == 1.0
    assert config.replay.stub_tools is True
    assert config.per_scenario_budget.max_turns == 15
