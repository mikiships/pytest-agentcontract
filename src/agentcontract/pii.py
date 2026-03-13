"""PII scanning for recorded agent trajectory cassettes."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentcontract.types import AgentRun, ToolCall, Turn


@dataclass(frozen=True)
class PiiFinding:
    """A single potential PII finding within a cassette."""

    detector: str
    severity: str
    confidence: str
    turn_index: int
    role: str
    field_path: str
    preview: str
    tool_function: str | None = None


@dataclass(frozen=True)
class PiiScanReport:
    """Structured output for a PII scan."""

    findings: tuple[PiiFinding, ...]

    @property
    def total_findings(self) -> int:
        """Return the total number of findings."""
        return len(self.findings)

    @property
    def detector_counts(self) -> dict[str, int]:
        """Return finding counts grouped by detector."""
        counts = Counter(finding.detector for finding in self.findings)
        return {
            detector.name: counts[detector.name]
            for detector in DETECTORS
            if counts[detector.name]
        }


@dataclass(frozen=True)
class _Detector:
    name: str
    pattern: re.Pattern[str]
    severity: str
    confidence: str
    validator: Callable[[str], bool] | None = None
    redactor: Callable[[str], str] | None = None


def _is_valid_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return 10 <= len(digits) <= 15 and any(char in candidate for char in "()+-. ")


def _is_valid_ssn(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return len(digits) == 9 and digits != "000000000"


def _passes_luhn(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 13 <= len(digits) <= 19:
        return False
    if len(set(digits)) == 1:
        return False

    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        value = int(digit)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _redact_email(candidate: str) -> str:
    local_part, _, domain = candidate.partition("@")
    if not domain:
        return _redact_digits(candidate)

    labels = domain.split(".")
    masked_labels: list[str] = []
    for index, label in enumerate(labels):
        if not label:
            masked_labels.append(label)
            continue
        if index == len(labels) - 1:
            masked_labels.append(label)
            continue
        masked_labels.append(label[0] + "*" * max(len(label) - 1, 0))

    return (
        local_part[:1]
        + "*" * max(len(local_part) - 1, 0)
        + "@"
        + ".".join(masked_labels)
    )


def _redact_digits(candidate: str) -> str:
    digit_total = sum(char.isdigit() for char in candidate)
    visible_digits = 4 if digit_total > 4 else 1
    masked_seen = 0
    result: list[str] = []
    for char in candidate:
        if not char.isdigit():
            result.append(char)
            continue

        masked_seen += 1
        if masked_seen <= digit_total - visible_digits:
            result.append("*")
        else:
            result.append(char)
    return "".join(result)


DETECTORS: tuple[_Detector, ...] = (
    _Detector(
        name="email",
        pattern=re.compile(
            r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
            re.IGNORECASE,
        ),
        severity="high",
        confidence="high",
        redactor=_redact_email,
    ),
    _Detector(
        name="phone",
        pattern=re.compile(
            r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\w)"
        ),
        severity="medium",
        confidence="medium",
        validator=_is_valid_phone,
        redactor=_redact_digits,
    ),
    _Detector(
        name="ssn",
        pattern=re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)"),
        severity="high",
        confidence="high",
        validator=_is_valid_ssn,
        redactor=_redact_digits,
    ),
    _Detector(
        name="payment_card",
        pattern=re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
        severity="high",
        confidence="medium",
        validator=_passes_luhn,
        redactor=_redact_digits,
    ),
)


def scan_run_for_pii(run: AgentRun) -> PiiScanReport:
    """Scan an ``AgentRun`` for high-signal PII indicators."""
    findings: list[PiiFinding] = []

    for turn in run.turns:
        _scan_turn(findings, turn)

    findings.sort(
        key=lambda finding: (
            finding.turn_index,
            finding.tool_function or "",
            finding.field_path,
            finding.detector,
            finding.preview,
        )
    )
    return PiiScanReport(findings=tuple(findings))


def _scan_turn(findings: list[PiiFinding], turn: Turn) -> None:
    role = turn.role.value
    if turn.content:
        _scan_scalar(
            findings,
            value=turn.content,
            turn_index=turn.index,
            role=role,
            field_path="content",
            tool_function=None,
        )

    for tool_call_index, tool_call in enumerate(turn.tool_calls):
        _scan_tool_call(
            findings,
            turn_index=turn.index,
            role=role,
            tool_call=tool_call,
            tool_call_index=tool_call_index,
        )


def _scan_tool_call(
    findings: list[PiiFinding],
    *,
    turn_index: int,
    role: str,
    tool_call: ToolCall,
    tool_call_index: int,
) -> None:
    arguments_path = f"tool_calls[{tool_call_index}].arguments"
    _scan_value(
        findings,
        value=tool_call.arguments,
        turn_index=turn_index,
        role=role,
        field_path=arguments_path,
        tool_function=tool_call.function,
    )

    result_path = f"tool_calls[{tool_call_index}].result"
    _scan_value(
        findings,
        value=tool_call.result,
        turn_index=turn_index,
        role=role,
        field_path=result_path,
        tool_function=tool_call.function,
    )


def _scan_value(
    findings: list[PiiFinding],
    *,
    value: Any,
    turn_index: int,
    role: str,
    field_path: str,
    tool_function: str | None,
) -> None:
    _scan_scalar(
        findings,
        value=value,
        turn_index=turn_index,
        role=role,
        field_path=field_path,
        tool_function=tool_function,
    )

    if isinstance(value, dict):
        for key, nested_value in value.items():
            _scan_value(
                findings,
                value=nested_value,
                turn_index=turn_index,
                role=role,
                field_path=f"{field_path}.{key}",
                tool_function=tool_function,
            )
        return

    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            _scan_value(
                findings,
                value=nested_value,
                turn_index=turn_index,
                role=role,
                field_path=f"{field_path}[{index}]",
                tool_function=tool_function,
            )


def _scan_scalar(
    findings: list[PiiFinding],
    *,
    value: Any,
    turn_index: int,
    role: str,
    field_path: str,
    tool_function: str | None,
) -> None:
    if isinstance(value, bool) or value is None:
        return

    if not isinstance(value, (str, int)):
        return

    text = str(value)
    for detector in DETECTORS:
        for match in detector.pattern.finditer(text):
            candidate = match.group(0)
            if detector.validator is not None and not detector.validator(candidate):
                continue

            findings.append(
                PiiFinding(
                    detector=detector.name,
                    severity=detector.severity,
                    confidence=detector.confidence,
                    turn_index=turn_index,
                    role=role,
                    field_path=field_path,
                    preview=_make_preview(text, match.start(), match.end(), detector),
                    tool_function=tool_function or None,
                )
            )


def _make_preview(text: str, start: int, end: int, detector: _Detector) -> str:
    snippet_start = max(0, start - 16)
    snippet_end = min(len(text), end + 16)
    prefix = _normalize_preview_text(text[snippet_start:start])
    suffix = _normalize_preview_text(text[end:snippet_end])
    redactor = detector.redactor or _redact_digits
    masked = redactor(text[start:end])
    preview = f"{prefix}{masked}{suffix}"
    if snippet_start > 0:
        preview = f"...{preview}"
    if snippet_end < len(text):
        preview = f"{preview}..."
    return preview


def _normalize_preview_text(value: str) -> str:
    return re.sub(r"\s+", " ", value)
