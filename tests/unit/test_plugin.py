"""Tests for pytest plugin helpers."""

from pathlib import Path

import pytest

from agentcontract.plugin import _resolve_cassette_path


def test_resolve_cassette_path_for_simple_scenario(tmp_path: Path) -> None:
    path = _resolve_cassette_path(tmp_path, "refund-eligible")

    assert path == (tmp_path / "refund-eligible.agentrun.json").resolve(strict=False)


def test_resolve_cassette_path_allows_nested_scenario_inside_base(
    tmp_path: Path,
) -> None:
    path = _resolve_cassette_path(tmp_path, "customer-support/refund-eligible")

    assert path == (
        tmp_path / "customer-support" / "refund-eligible.agentrun.json"
    ).resolve(strict=False)


def test_resolve_cassette_path_rejects_parent_directory_traversal(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cassette path must stay within"):
        _resolve_cassette_path(tmp_path, "customer-support/../../escape")


def test_resolve_cassette_path_rejects_absolute_scenario_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute paths are not allowed"):
        _resolve_cassette_path(tmp_path, str(tmp_path / "escape"))
