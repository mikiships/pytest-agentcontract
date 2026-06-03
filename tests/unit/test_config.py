"""Tests for config parsing."""

from pathlib import Path

from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AgentContractConfig
from agentcontract.types import AgentRun, RunMetadata, ToolCall, Turn, TurnRole


def test_config_from_dict():
    raw = {
        "version": "1",
        "scenarios": {"include": ["tests/**/*.agentrun.json"]},
        "replay": {"stub_tools": True},
        "defaults": {
            "assertions": [{"type": "contains", "target": "final_response", "value": "refund"}]
        },
        "overrides": {
            "denied-flow": {"assertions": [{"type": "not_called", "target": "tool:process_refund"}]}
        },
        "policies": [{"name": "tools", "type": "tool_allowlist", "tools": ["lookup_order"]}],
    }

    config = AgentContractConfig.from_dict(raw)
    assert config.version == "1"
    assert config.scenario_include == ["tests/**/*.agentrun.json"]
    assert config.replay.stub_tools is True
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
    assert config.version == "1"
    assert config.scenario_include == ["tests/scenarios/**/*.agentrun.json"]
    assert config.replay.stub_tools is True


def test_config_from_dict_handles_null_sections():
    config = AgentContractConfig.from_dict(
        {
            "scenarios": None,
            "replay": None,
            "defaults": None,
            "overrides": None,
            "policies": None,
        }
    )
    assert config.scenario_include == ["tests/scenarios/**/*.agentrun.json"]
    assert config.default_assertions == []
    assert config.overrides == {}
    assert config.policies == []
    assert config.replay.stub_tools is True


def test_config_from_dict_coerces_null_policy_lists():
    config = AgentContractConfig.from_dict(
        {"policies": [{"name": "tools", "type": "tool_allowlist", "tools": None}]}
    )
    assert config.policies[0].tools == []

    run = AgentRun(
        metadata=RunMetadata(scenario="null-policy-lists"),
        turns=[
            Turn(
                index=0,
                role=TurnRole.ASSISTANT,
                tool_calls=[ToolCall(id="tc1", function="lookup_order", arguments={})],
            )
        ],
    )
    result = AssertionEngine().check(run, policies=config.policies)
    assert not result.passed


def test_config_from_dict_coerces_scalar_types():
    config = AgentContractConfig.from_dict(
        {
            "version": 2,
            "replay": {"stub_tools": "false"},
        }
    )

    assert config.version == "2"
    assert config.replay.stub_tools is False


def test_config_from_dict_ignores_irrelevant_extra_sections():
    config = AgentContractConfig.from_dict(
        {
            "scenarios": {
                "include": ["custom/**/*.agentrun.json"],
                "exclude": ["ignored/**/*.agentrun.json"],
            },
            "replay": {"stub_tools": False, "model": "ignored-model", "concurrency": 99},
            "defaults": {
                "assertions": [
                    {
                        "type": "contains",
                        "target": "final_response",
                        "value": "approved",
                        "threshold": 0.9,
                        "prompt": "ignored",
                    }
                ]
            },
            "policies": [
                {
                    "name": "tools",
                    "type": "tool_allowlist",
                    "tools": ["lookup_order"],
                    "block": ["process_refund"],
                }
            ],
            "thresholds": {"suite_pass_rate": "0.75"},
            "budgets": {"per_scenario": {"max_turns": "9"}},
            "reporting": {"github_comment": False},
            "baseline": {"branch": "main"},
        }
    )

    assert config.scenario_include == ["custom/**/*.agentrun.json"]
    assert config.replay.stub_tools is False
    assert len(config.default_assertions) == 1
    assert config.default_assertions[0].type == "contains"
    assert config.default_assertions[0].target == "final_response"
    assert config.default_assertions[0].value == "approved"
    assert config.default_assertions[0].schema is None
    assert config.policies[0].tools == ["lookup_order"]


def test_discover_accepts_file_path_start(tmp_path: Path):
    project_dir = tmp_path / "project"
    nested_dir = project_dir / "pkg"
    nested_dir.mkdir(parents=True)

    config_file = project_dir / "agentcontract.yml"
    config_file.write_text('version: "2"\n')
    module_file = nested_dir / "test_module.py"
    module_file.write_text("# test module\n")

    config = AgentContractConfig.discover(start=module_file)

    assert config.version == "2"
