"""Scanner behavior and output confidentiality."""

import json
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
