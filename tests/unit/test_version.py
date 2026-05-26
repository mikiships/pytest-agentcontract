"""Tests for package version metadata."""

from pathlib import Path

import pytest

import agentcontract

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility
    tomllib = pytest.importorskip("tomli")


def test_package_version_matches_project_metadata() -> None:
    project_root = Path(__file__).resolve().parents[2]
    project_metadata = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert agentcontract.__version__ == project_metadata["project"]["version"]
