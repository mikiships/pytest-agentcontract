"""Tests for config parsing."""

from pathlib import Path

from agentcontract.assertions.engine import AssertionEngine
from agentcontract.config import AgentContractConfig
from agentcontract.types import AgentRun, RunMetadata, ToolCall, Turn, TurnRole


def test_config_from_dict():
    raw = {
        "version": "1",
        "scenarios": {"include": ["tests/**/*.agentrun.json"]},
        "replay": {"stub_tools": True, "concurrency": 3},
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


def test_config_from_dict_handles_null_sections():
    config = AgentContractConfig.from_dict(
        {
            "scenarios": None,
            "replay": None,
            "defaults": None,
            "overrides": None,
            "policies": None,
            "thresholds": None,
            "budgets": None,
            "reporting": None,
            "baseline": None,
        }
    )
    assert config.scenario_include == ["tests/scenarios/**/*.agentrun.json"]
    assert config.scenario_exclude == []
    assert config.default_assertions == []
    assert config.overrides == {}
    assert config.policies == []
    assert config.replay.stub_tools is True


def test_config_from_dict_coerces_null_policy_lists():
    config = AgentContractConfig.from_dict(
        {"policies": [{"name": "tools", "type": "tool_allowlist", "tools": None, "block": None}]}
    )
    assert config.policies[0].tools == []
    assert config.policies[0].block == []

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
            "replay": {
                "model": 123,
                "seed": "7",
                "stub_tools": "false",
                "concurrency": "4",
            },
            "thresholds": {"suite_pass_rate": "0.75"},
            "budgets": {
                "per_scenario": {
                    "max_cost_usd": "0.15",
                    "max_latency_ms": "1200",
                    "max_turns": "9",
                },
                "suite": {"max_cost_usd": "3.5"},
            },
            "baseline": {"branch": 9, "show_deltas": "0"},
            "reporting": {"github_comment": "no", "artifact_path": 42},
        }
    )

    assert config.version == "2"
    assert config.replay.model == "123"
    assert config.replay.seed == 7
    assert config.replay.stub_tools is False
    assert config.replay.concurrency == 4
    assert config.suite_pass_rate == 0.75
    assert config.per_scenario_budget.max_cost_usd == 0.15
    assert config.per_scenario_budget.max_latency_ms == 1200.0
    assert config.per_scenario_budget.max_turns == 9
    assert config.suite_budget_usd == 3.5
    assert config.baseline_branch == "9"
    assert config.show_deltas is False
    assert config.github_comment is False
    assert config.artifact_path == "42"


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
