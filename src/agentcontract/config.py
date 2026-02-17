"""Configuration loader for agentcontract.yml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ReplayConfig:
    model: str = ""
    seed: int | None = 42
    stub_tools: bool = True
    concurrency: int = 5


@dataclass
class BudgetConfig:
    max_cost_usd: float = 0.05
    max_latency_ms: float = 10000
    max_turns: int = 15


@dataclass
class AssertionSpec:
    """A single assertion definition from config."""

    type: str
    target: str = ""
    value: str | None = None
    threshold: float | None = None
    prompt: str | None = None
    schema: dict[str, Any] | None = None
    judge_model: str | None = None
    tools: list[str] | None = None
    block: list[str] | None = None


@dataclass
class PolicySpec:
    """A policy definition from config."""

    name: str
    type: str
    target: str = ""
    tools: list[str] = field(default_factory=list)
    block: list[str] = field(default_factory=list)


@dataclass
class ScenarioOverride:
    """Per-scenario assertion overrides."""

    assertions: list[AssertionSpec] = field(default_factory=list)


@dataclass
class AgentContractConfig:
    """Parsed agentcontract.yml configuration."""

    version: str = "1"
    scenario_include: list[str] = field(
        default_factory=lambda: ["tests/scenarios/**/*.agentrun.json"]
    )
    scenario_exclude: list[str] = field(default_factory=list)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    default_assertions: list[AssertionSpec] = field(default_factory=list)
    overrides: dict[str, ScenarioOverride] = field(default_factory=dict)
    policies: list[PolicySpec] = field(default_factory=list)
    suite_pass_rate: float = 1.0
    per_scenario_budget: BudgetConfig = field(default_factory=BudgetConfig)
    suite_budget_usd: float = 2.0
    baseline_branch: str = "main"
    show_deltas: bool = True
    github_comment: bool = True
    artifact_path: str = "agentci-results/"

    @classmethod
    def from_file(cls, path: Path) -> AgentContractConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentContractConfig:
        """Parse a raw dict into config."""
        raw = _coerce_dict(raw)
        scenarios = _coerce_dict(raw.get("scenarios"))
        replay_raw = _coerce_dict(raw.get("replay"))
        defaults_raw = _coerce_dict(raw.get("defaults"))
        budgets = _coerce_dict(raw.get("budgets"))
        per_scenario = _coerce_dict(budgets.get("per_scenario"))
        suite = _coerce_dict(budgets.get("suite"))
        reporting = _coerce_dict(raw.get("reporting"))
        baseline = _coerce_dict(raw.get("baseline"))
        thresholds = _coerce_dict(raw.get("thresholds"))

        default_assertions = [
            _parse_assertion(a)
            for a in _coerce_list(defaults_raw.get("assertions"), [])
            if isinstance(a, dict)
        ]

        overrides: dict[str, ScenarioOverride] = {}
        for name, override_raw in _coerce_dict(raw.get("overrides")).items():
            override_raw = _coerce_dict(override_raw)
            overrides[name] = ScenarioOverride(
                assertions=[
                    _parse_assertion(a)
                    for a in _coerce_list(override_raw.get("assertions"), [])
                    if isinstance(a, dict)
                ]
            )

        policies = [
            _parse_policy(p)
            for p in _coerce_list(raw.get("policies"), [])
            if isinstance(p, dict)
        ]

        return cls(
            version=_coerce_str(raw.get("version"), "1"),
            scenario_include=_coerce_list(
                scenarios.get("include"), ["tests/scenarios/**/*.agentrun.json"]
            ),
            scenario_exclude=_coerce_list(scenarios.get("exclude"), []),
            replay=ReplayConfig(
                model=_coerce_str(replay_raw.get("model"), ""),
                seed=_coerce_optional_int(replay_raw.get("seed"), 42),
                stub_tools=_coerce_bool(replay_raw.get("stub_tools"), True),
                concurrency=_coerce_int(replay_raw.get("concurrency"), 5),
            ),
            default_assertions=default_assertions,
            overrides=overrides,
            policies=policies,
            suite_pass_rate=_coerce_float(thresholds.get("suite_pass_rate"), 1.0),
            per_scenario_budget=BudgetConfig(
                max_cost_usd=_coerce_float(per_scenario.get("max_cost_usd"), 0.05),
                max_latency_ms=_coerce_float(per_scenario.get("max_latency_ms"), 10000),
                max_turns=_coerce_int(per_scenario.get("max_turns"), 15),
            ),
            suite_budget_usd=_coerce_float(suite.get("max_cost_usd"), 2.0),
            baseline_branch=_coerce_str(baseline.get("branch"), "main"),
            show_deltas=_coerce_bool(baseline.get("show_deltas"), True),
            github_comment=_coerce_bool(reporting.get("github_comment"), True),
            artifact_path=_coerce_str(reporting.get("artifact_path"), "agentci-results/"),
        )

    @classmethod
    def discover(cls, start: Path | None = None) -> AgentContractConfig:
        """Walk up from start (or cwd) looking for agentcontract.yml."""
        search = start or Path.cwd()
        if search.is_file():
            search = search.parent
        for directory in [search, *search.parents]:
            candidate = directory / "agentcontract.yml"
            if candidate.exists():
                return cls.from_file(candidate)
        return cls()  # defaults


def _parse_assertion(raw: dict[str, Any]) -> AssertionSpec:
    return AssertionSpec(
        type=raw["type"],
        target=raw.get("target", ""),
        value=raw.get("value"),
        threshold=raw.get("threshold"),
        prompt=raw.get("prompt"),
        schema=raw.get("schema"),
        judge_model=raw.get("judge_model"),
        tools=raw.get("tools"),
        block=raw.get("block"),
    )


def _parse_policy(raw: dict[str, Any]) -> PolicySpec:
    return PolicySpec(
        name=raw["name"],
        type=raw["type"],
        target=raw.get("target", ""),
        tools=[str(item) for item in _coerce_list(raw.get("tools"), []) if item is not None],
        block=[str(item) for item in _coerce_list(raw.get("block"), []) if item is not None],
    )


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Normalize config sections that may be null in YAML."""
    if isinstance(value, dict):
        return value
    return {}


def _coerce_list(value: Any, default: list[Any]) -> list[Any]:
    """Normalize config list fields while preserving sensible defaults."""
    if isinstance(value, list):
        return value
    return list(default)


def _coerce_str(value: Any, default: str = "") -> str:
    """Normalize scalar fields to strings while preserving defaults for nulls."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_bool(value: Any, default: bool) -> bool:
    """Normalize booleans from YAML-compatible scalar values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    """Normalize numeric config fields that should be integers."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Any, default: int | None = None) -> int | None:
    """Normalize optional integer config fields."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    """Normalize numeric config fields that should be floats."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
