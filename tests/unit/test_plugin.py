"""Tests for plugin helpers."""

from pathlib import Path

from agentcontract.plugin import _cassette_path, _scenario_filename


def test_scenario_filename_preserves_safe_names() -> None:
    assert _scenario_filename("refund-eligible") == "refund-eligible"


def test_scenario_filename_hashes_sanitized_names() -> None:
    first = _scenario_filename("refund eligible")
    second = _scenario_filename("refund/eligible")

    assert first.startswith("refund-eligible-")
    assert second.startswith("refund-eligible-")
    assert first != second


def test_cassette_path_keeps_cassettes_inside_scenarios_dir(tmp_path: Path) -> None:
    scenarios_dir = tmp_path / "scenarios"

    cassette_path = _cassette_path(scenarios_dir, "../../secrets/production")

    assert cassette_path.parent == scenarios_dir
    assert cassette_path.relative_to(scenarios_dir) == Path(cassette_path.name)
    assert ".." not in cassette_path.name
    assert "/" not in cassette_path.name
    assert "\\" not in cassette_path.name
