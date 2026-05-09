"""PII scanning helpers for agentcontract cassette files."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PiiFinding:
    """A masked PII finding in a cassette file."""

    path: Path
    json_path: str
    kind: str
    preview: str


class PiiScanError(Exception):
    """Raised when a cassette cannot be loaded for scanning."""

    def __init__(self, path: Path, error: BaseException) -> None:
        self.path = path
        self.error = error
        super().__init__(
            f"failed to read cassette '{path}' ({type(error).__name__}): {error}"
        )


@dataclass(frozen=True)
class _Detector:
    kind: str
    pattern: re.Pattern[str]
    mask: Callable[[str], str]
    validate: Callable[[str], bool] | None = None


@dataclass(frozen=True)
class _PiiMatch:
    start: int
    end: int
    kind: str
    preview: str


_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9_%+-])"
)
_SSN_RE = re.compile(
    r"(?<!\d)(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)"
)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+1[2-9]\d{2}[2-9]\d{6}|(?:\+?1[\s.-]?)?"
    r"(?:\([2-9]\d{2}\)[\s.-]?|[2-9]\d{2}[\s.-])"
    r"[2-9]\d{2}[\s.-]\d{4})(?!\d)"
)
_CREDIT_CARD_RE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?![ -]?\d)")
_JSON_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def scan_file(path: Path) -> list[PiiFinding]:
    """Scan a cassette file for high-confidence PII patterns."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PiiScanError(path, exc) from exc

    return scan_data(data, path)


def scan_data(data: Any, path: Path | str = Path("<memory>")) -> list[PiiFinding]:
    """Scan raw cassette data for high-confidence PII patterns."""
    cassette_path = Path(path)
    findings: list[PiiFinding] = []
    for value, json_path in _iter_scan_roots(data):
        _scan_value(cassette_path, json_path, value, findings)
    return findings


def _iter_scan_roots(data: Any) -> Iterable[tuple[Any, str]]:
    if not isinstance(data, dict):
        return

    turns = data.get("turns")
    if not isinstance(turns, list):
        return

    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue

        turn_path = f"$.turns[{turn_index}]"
        if "content" in turn:
            yield turn["content"], f"{turn_path}.content"

        tool_calls = turn.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue

            call_path = f"{turn_path}.tool_calls[{call_index}]"
            if "arguments" in tool_call:
                yield tool_call["arguments"], f"{call_path}.arguments"
            if "result" in tool_call:
                yield tool_call["result"], f"{call_path}.result"


def _scan_value(
    cassette_path: Path,
    json_path: str,
    value: Any,
    findings: list[PiiFinding],
) -> None:
    if isinstance(value, str):
        _scan_text(cassette_path, json_path, value, findings, _DETECTORS)
        return

    if isinstance(value, int) and not isinstance(value, bool):
        _scan_text(cassette_path, json_path, str(value), findings, [_CREDIT_CARD_DETECTOR])
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_value(cassette_path, f"{json_path}[{index}]", item, findings)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_path = _json_child_path(json_path, key_text)
            _scan_text(cassette_path, child_path, key_text, findings, _DETECTORS)
            _scan_value(cassette_path, child_path, item, findings)


def _scan_text(
    cassette_path: Path,
    json_path: str,
    value: str,
    findings: list[PiiFinding],
    detectors: Iterable[_Detector],
) -> None:
    for match in _find_pii_matches(value, detectors):
        findings.append(
            PiiFinding(
                path=cassette_path,
                json_path=json_path,
                kind=match.kind,
                preview=match.preview,
            )
        )


def _find_pii_matches(value: str, detectors: Iterable[_Detector]) -> list[_PiiMatch]:
    matches: list[_PiiMatch] = []
    for detector in detectors:
        for match in detector.pattern.finditer(value):
            matched_value = match.group(0)
            if detector.validate is not None and not detector.validate(matched_value):
                continue
            matches.append(
                _PiiMatch(
                    start=match.start(),
                    end=match.end(),
                    kind=detector.kind,
                    preview=detector.mask(matched_value),
                )
            )

    return sorted(matches, key=lambda item: (item.start, item.end, item.kind))


def _json_child_path(parent: str, key: str) -> str:
    safe_key = _mask_text_for_path(key)
    if safe_key == key and _JSON_IDENTIFIER_RE.match(key):
        return f"{parent}.{key}"
    return f"{parent}[{json.dumps(safe_key)}]"


def _mask_text_for_path(value: str) -> str:
    matches = sorted(
        _find_pii_matches(value, _DETECTORS),
        key=lambda item: (item.start, -(item.end - item.start)),
    )
    if not matches:
        return value

    chunks: list[str] = []
    position = 0
    for match in matches:
        if match.start < position:
            continue
        chunks.append(value[position : match.start])
        chunks.append(match.preview)
        position = match.end
    chunks.append(value[position:])
    return "".join(chunks)


def _mask_email(value: str) -> str:
    local, domain = value.split("@", 1)
    domain_name, _, tld = domain.rpartition(".")
    masked_local = f"{local[:1]}***"
    masked_domain = f"{domain_name[:1]}***" if domain_name else "***"
    return f"{masked_local}@{masked_domain}.{tld}" if tld else f"{masked_local}@{masked_domain}"


def _mask_last4(value: str, prefix: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 4:
        return prefix
    return f"{prefix}{digits[-4:]}"


def _is_luhn_valid(value: str) -> bool:
    digits = "".join(ch for ch in value if ch.isdigit())
    if not 13 <= len(digits) <= 19:
        return False
    if len(set(digits)) == 1:
        return False

    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        number = int(digit)
        if index % 2 == parity:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


_CREDIT_CARD_DETECTOR = _Detector(
    kind="credit_card",
    pattern=_CREDIT_CARD_RE,
    mask=lambda value: _mask_last4(value, "**** **** **** "),
    validate=_is_luhn_valid,
)
_DETECTORS = [
    _Detector(kind="email", pattern=_EMAIL_RE, mask=_mask_email),
    _Detector(kind="ssn", pattern=_SSN_RE, mask=lambda value: _mask_last4(value, "***-**-")),
    _Detector(kind="phone", pattern=_PHONE_RE, mask=lambda value: _mask_last4(value, "***-***-")),
    _CREDIT_CARD_DETECTOR,
]
