"""Tests for static test-gap inspection."""

from __future__ import annotations

from pathlib import Path

from agentcontract.test_gap import find_test_gaps, format_test_gap_report


def test_find_test_gaps_ignores_init_files_and_orders_results(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _write(repo / "src/agentcontract/__init__.py", "")
    _write(repo / "src/agentcontract/zeta.py", "VALUE = 1\n")
    _write(repo / "src/agentcontract/alpha.py", "VALUE = 1\n")
    _write(repo / "tests/unit/test_zeta.py", "from agentcontract.cli import main\n")

    report = find_test_gaps(repo)

    assert report.scanned_modules == ("agentcontract.alpha", "agentcontract.zeta")
    assert [gap.module_name for gap in report.missing] == ["agentcontract.alpha"]


def test_find_test_gaps_matches_nested_modules_to_existing_test_layout(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _write(repo / "src/agentcontract/replay/__init__.py", "")
    _write(repo / "src/agentcontract/replay/engine.py", "def run() -> None:\n    return None\n")
    _write(repo / "src/agentcontract/recorder/__init__.py", "")
    _write(
        repo / "src/agentcontract/recorder/interceptors.py",
        "def patch() -> None:\n    return None\n",
    )
    _write(repo / "tests/unit/test_replay.py", "def test_replay() -> None:\n    assert True\n")
    _write(
        repo / "tests/unit/test_interceptors.py",
        "def test_interceptors() -> None:\n    assert True\n",
    )

    report = find_test_gaps(repo)

    assert not report.missing
    assert [gap.module_name for gap in report.weak] == []


def test_find_test_gaps_flags_shared_package_tests_and_hotspots(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _write(repo / "src/agentcontract/adapters/__init__.py", "")
    _write(
        repo / "src/agentcontract/adapters/langgraph.py",
        "def record() -> None:\n    return None\n",
    )
    _write(
        repo / "src/agentcontract/adapters/openai_agents.py",
        "def record() -> None:\n    return None\n",
    )
    _write(
        repo / "src/agentcontract/cli.py",
        '\n'.join(
            [
                'subparsers.add_parser("info")',
                'subparsers.add_parser("validate")',
                'subparsers.add_parser("init")',
                'subparsers.add_parser("test-gap")',
                "",
            ]
        ),
    )
    _write(
        repo / "src/agentcontract/plugin.py",
        "def pytest_addoption() -> None:\n    return None\n",
    )
    _write(repo / "tests/unit/test_adapters.py", "def test_adapters() -> None:\n    assert True\n")
    _write(
        repo / "tests/unit/test_cli.py",
        'from agentcontract.cli import main\n\n'
        'def test_cli() -> None:\n'
        '    assert main(["info", "cassette.agentrun.json"]) == 0\n',
    )

    report = find_test_gaps(repo)

    assert [gap.module_name for gap in report.missing] == ["agentcontract.plugin"]
    weak_notes = {gap.module_name: gap.notes for gap in report.weak}
    assert "agentcontract.adapters.langgraph" in weak_notes
    assert "agentcontract.adapters.openai_agents" in weak_notes
    assert weak_notes["agentcontract.cli"] == (
        "CLI subcommands without explicit test hits: init, test-gap, validate",
    )


def test_find_test_gaps_does_not_flag_plugin_with_dedicated_test_file(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _write(
        repo / "src/agentcontract/plugin.py",
        "def pytest_addoption() -> None:\n    return None\n",
    )
    _write(repo / "tests/unit/test_plugin.py", "def test_plugin() -> None:\n    assert True\n")

    report = find_test_gaps(repo)

    assert report.missing == ()
    assert all(gap.module_name != "agentcontract.plugin" for gap in report.weak)


def test_find_test_gaps_requires_explicit_cli_invocation_hits(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _write(
        repo / "src/agentcontract/cli.py",
        '\n'.join(
            [
                'subparsers.add_parser("info")',
                'subparsers.add_parser("validate")',
                'subparsers.add_parser("init")',
                'subparsers.add_parser("test-gap")',
                "",
            ]
        ),
    )
    _write(
        repo / "tests/unit/test_cli.py",
        'from agentcontract.cli import main\n\n'
        'def test_cli_notes() -> None:\n'
        '    note = "info"\n'
        '    assert note == "info"\n',
    )

    report = find_test_gaps(repo)

    weak_notes = {gap.module_name: gap.notes for gap in report.weak}
    assert weak_notes["agentcontract.cli"] == (
        "CLI subcommands without explicit test hits: info, init, test-gap, validate",
    )


def test_format_test_gap_report_includes_limitation_note(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _write(
        repo / "src/agentcontract/plugin.py",
        "def pytest_addoption() -> None:\n    return None\n",
    )

    report = find_test_gaps(repo)
    output = format_test_gap_report(report)

    assert "Missing coverage:" in output
    assert "v1 limitation" in output


def _make_repo(root: Path) -> Path:
    (root / "src/agentcontract").mkdir(parents=True)
    (root / "tests/unit").mkdir(parents=True)
    return root


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
