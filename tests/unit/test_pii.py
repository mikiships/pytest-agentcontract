"""Scanner behavior and output confidentiality."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentcontract.cli import main
from agentcontract.pii import scan_value


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("review@example.invalid", "email"),
        ("(202) 555-0100", "us_phone"),
        ("+1 202-555-0100", "us_phone"),
        ("202.555.0100", "us_phone"),
        ("000-00-0000", "us_ssn"),
    ],
)
def test_detectors(text, category):
    assert list(scan_value(text)) == [("$", category)]


@pytest.mark.parametrize(
    "text",
    [
        "2025550100",
        "1202-555-01000",
        "x000-00-0000y",
        "2026-10-11",
        "a@localhost",
        "1.2.3",
        "",
        None,
        123,
    ],
)
def test_boundaries(text):
    assert list(scan_value(text)) == []


def test_nested_keys_and_values():
    value = {"private name": [{"review@example.invalid": {"result": "000-00-0000"}}]}
    assert list(scan_value(value)) == [
        ("$.entry[0].value[0].entry[0].key", "email"),
        ("$.entry[0].value[0].entry[0].value.entry[0].value", "us_ssn"),
    ]


def test_cli_sanitized_deterministic_read_only(tmp_path, capsys):
    folder = tmp_path / "private person"
    folder.mkdir()
    cassette = folder / "review@example.invalid.agentrun.json"
    original = '{"private name":"000-00-0000","private name":"(202) 555-0100"}'
    cassette.write_text(original)
    (tmp_path / "a.agentrun.json").write_text("{}")
    (tmp_path / "ignored.json").write_text("bad")
    assert main(["scan-pii", str(tmp_path)]) == 1
    first = capsys.readouterr()
    assert main(["scan-pii", str(tmp_path)]) == 1
    assert capsys.readouterr() == first
    report = json.loads(first.out)
    assert report["scanned_files"] == 2
    assert len(report["findings"]) == 2
    assert report["findings"][0]["file"] == "file-0002"
    for secret in ["private", "review@example.invalid", "000-00-0000", "(202) 555-0100"]:
        assert secret not in first.out + first.err
    assert cassette.read_text() == original


@pytest.mark.parametrize("content", ["NaN", "{private name", "\ud800", "[" * 2000])
def test_malformed_continues(tmp_path, capsys, content):
    (tmp_path / "bad.agentrun.json").write_bytes(content.encode("utf-8", errors="surrogatepass"))
    (tmp_path / "good.agentrun.json").write_text('"review@example.invalid"')
    assert main(["scan-pii", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["scanned_files"] == 1
    assert len(report["findings"]) == 1
    assert len(report["errors"]) == 1
    assert "private name" not in captured.out + captured.err
    assert "review@example.invalid" not in captured.out + captured.err


def test_unreadable_continues(tmp_path, capsys, monkeypatch):
    bad = tmp_path / "bad.agentrun.json"
    bad.write_text("{}")
    (tmp_path / "good.agentrun.json").write_text("{}")
    read = Path.read_text

    def denied(path, *args, **kwargs):
        if path == bad:
            raise PermissionError("private name")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    assert main(["scan-pii", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["scanned_files"] == 1
    assert "private name" not in captured.out + captured.err


def test_empty_missing_and_clean(tmp_path, capsys):
    assert main(["scan-pii", str(tmp_path)]) == 2
    assert main(["scan-pii", str(tmp_path / "private name")]) == 2
    cassette = tmp_path / "clean.json"
    cassette.write_text('{"turns":[],"metadata":{}}')
    assert main(["scan-pii", str(cassette)]) == 0
    captured = capsys.readouterr()
    assert "private name" not in captured.out + captured.err


def test_discovery_error(tmp_path, capsys, monkeypatch):
    def failed_walk(target, onerror):
        onerror(PermissionError("private name"))
        return iter([])

    monkeypatch.setattr("agentcontract.pii.os.walk", failed_walk)
    assert main(["scan-pii", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "discovery_failed" in captured.out
    assert "private name" not in captured.out + captured.err


def test_linked_directory_is_incomplete(tmp_path, capsys):
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    assert main(["scan-pii", str(tmp_path)]) == 2
    assert "discovery_failed" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["(202)5550100", "+1(202)5550100", "(202) 5550100"])
def test_parenthesized_phone_without_subscriber_separator(text):
    assert list(scan_value(text)) == [("$", "us_phone")]


@pytest.mark.parametrize("text", ["(202)55501000", "x(202)5550100", "(202)5550100x"])
def test_parenthesized_phone_boundaries(text):
    assert list(scan_value(text)) == []


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["clean.json", "review@example.invalid.agentrun.json"],
        ["--review@example.invalid"],
    ],
)
def test_invalid_cli_arguments_are_sanitized(arguments, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["scan-pii", *arguments])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "review@example.invalid" not in captured.out + captured.err
    assert "invalid command arguments" in captured.err



def test_compact_phone_cli_is_sanitized_and_read_only(tmp_path, capsys, monkeypatch):
    cassette = tmp_path / "(202)5550100.agentrun.json"
    original = b'{"nested": [{"phone": "(202)5550100"}]}'
    cassette.write_bytes(original)
    monkeypatch.setattr("sys.argv", ["agentcontract", "scan-pii", str(cassette)])
    assert main() == 1
    captured = capsys.readouterr()
    assert "(202)5550100" not in captured.out + captured.err
    assert json.loads(captured.out)["findings"] == [
        {
            "file": "file-0001",
            "location": "$.entry[0].value[0].entry[0].value",
            "category": "us_phone",
        }
    ]
    assert cassette.read_bytes() == original


@pytest.mark.parametrize(
    "arguments",
    [
        ["--private=review@example.invalid", "scan-pii", "clean.json"],
        ["--private", "scan-pii", "review@example.invalid"],
        ["review@example.invalid", "scan-pii"],
        ["--", "scan-pii", "clean.json", "review@example.invalid"],
        ["scan-pii", "--help=review@example.invalid"],
        ["--help=review@example.invalid", "scan-pii"],
        ["scan-pii", "-xreview@example.invalid"],
        ["scan-pii", "clean.json", "--private=review@example.invalid"],
    ],
)
@pytest.mark.parametrize("use_sys_argv", [False, True])
def test_parser_errors_never_echo_arguments(arguments, use_sys_argv, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["review@example.invalid", *arguments])
    with pytest.raises(SystemExit) as exc:
        main(None if use_sys_argv else arguments)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "review@example.invalid" not in captured.out + captured.err
    assert "invalid command arguments" in captured.err


@pytest.mark.parametrize("arguments", [["--help"], ["scan-pii", "--help"]])
def test_help_remains_available(arguments, capsys):
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 0
    assert "scan-pii" in capsys.readouterr().out


@pytest.mark.parametrize("prefix", [[], ["--private=review@example.invalid"]])
def test_module_entrypoint_sanitizes_argument_errors(prefix):
    result = subprocess.run(
        [sys.executable, "-m", "agentcontract.cli", *prefix,
         "scan-pii", "clean.json", "review@example.invalid"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "review@example.invalid" not in result.stdout + result.stderr
    assert "invalid command arguments" in result.stderr
