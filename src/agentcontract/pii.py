"""Minimal PII exposure scanner for recorded agent trajectories."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentcontract.types import AgentRun

SUPPORTED_CATEGORIES: tuple[str, ...] = ("email", "phone", "ssn", "credit_card")

_CATEGORY_ALIASES = {
    "credit-card": "credit_card",
    "credit_card_number": "credit_card",
    "credit-card-number": "credit_card",
    "phone_number": "phone",
    "phone-number": "phone",
    "us_phone": "phone",
    "us-phone": "phone",
}

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)
_PHONE_RE = re.compile(
    r"(?<!\w)"
    r"(?:\+?1[\s.-]?)?"
    r"(?:\([2-9]\d{2}\)\s*|[2-9]\d{2}[\s.-]+)"
    r"[2-9]\d{2}[\s.-]+\d{4}"
    r"(?!\w)"
)
_SSN_RE = re.compile(
    r"(?<!\d)"
    r"(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}"
    r"(?!\d)"
)
_CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_LOCATION_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class PiiFinding:
    """A safe-to-display PII finding."""

    category: str
    location: str
    snippet: str
    cassette_path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Serialize the finding without raw PII values."""
        return {
            "category": self.category,
            "location": self.location,
            "snippet": self.snippet,
            "cassette_path": self.cassette_path,
        }


@dataclass
class PiiScanResult:
    """Result of scanning one or more agent trajectories."""

    findings: list[PiiFinding] = field(default_factory=list)
    scanned_files: list[str] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        """Whether the scan found any PII."""
        return bool(self.findings)

    @property
    def finding_count(self) -> int:
        """Number of findings."""
        return len(self.findings)

    def extend(self, other: PiiScanResult) -> None:
        """Merge another scan result into this result."""
        self.findings.extend(other.findings)
        self.scanned_files.extend(other.scanned_files)

    def to_dict(self) -> dict[str, object]:
        """Serialize the result without raw PII values."""
        return {
            "finding_count": self.finding_count,
            "scanned_files": self.scanned_files,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def scan_text(
    text: str,
    *,
    location: str = "",
    cassette_path: str | Path | None = None,
    categories: Iterable[str] | str | None = None,
) -> PiiScanResult:
    """Scan a text string for high-confidence PII patterns."""
    return _scan_text(
        text,
        location=location,
        cassette_path=_path_to_str(cassette_path),
        categories=_normalize_categories(categories),
    )


def scan_value(
    payload: Any,
    *,
    location: str = "",
    cassette_path: str | Path | None = None,
    categories: Iterable[str] | str | None = None,
) -> PiiScanResult:
    """Recursively scan dict/list/scalar payload keys and values."""
    return _scan_payload(
        payload,
        location=location,
        cassette_path=_path_to_str(cassette_path),
        categories=_normalize_categories(categories),
    )


def scan_payload(
    payload: Any,
    *,
    location: str = "",
    cassette_path: str | Path | None = None,
    categories: Iterable[str] | str | None = None,
) -> PiiScanResult:
    """Compatibility alias for scan_value."""
    return scan_value(
        payload,
        location=location,
        cassette_path=cassette_path,
        categories=categories,
    )


def scan_run(
    run: AgentRun,
    *,
    cassette_path: str | Path | None = None,
    categories: Iterable[str] | str | None = None,
) -> PiiScanResult:
    """Scan the PII-relevant surfaces of an AgentRun."""
    normalized_categories = _normalize_categories(categories)
    normalized_path = _path_to_str(cassette_path)
    result = PiiScanResult()

    result.extend(
        _scan_payload(
            run.metadata.scenario,
            location="metadata.scenario",
            cassette_path=normalized_path,
            categories=normalized_categories,
        )
    )
    result.extend(
        _scan_payload(
            run.metadata.description,
            location="metadata.description",
            cassette_path=normalized_path,
            categories=normalized_categories,
        )
    )
    result.extend(
        _scan_payload(
            run.metadata.tags,
            location="metadata.tags",
            cassette_path=normalized_path,
            categories=normalized_categories,
        )
    )

    for turn_index, turn in enumerate(run.turns):
        turn_location = f"turns[{turn_index}]"
        if turn.content is not None:
            result.extend(
                _scan_payload(
                    turn.content,
                    location=f"{turn_location}.content",
                    cassette_path=normalized_path,
                    categories=normalized_categories,
                )
            )

        for call_index, tool_call in enumerate(turn.tool_calls):
            call_location = f"{turn_location}.tool_calls[{call_index}]"
            result.extend(
                _scan_payload(
                    tool_call.function,
                    location=f"{call_location}.function",
                    cassette_path=normalized_path,
                    categories=normalized_categories,
                )
            )
            result.extend(
                _scan_payload(
                    tool_call.arguments,
                    location=f"{call_location}.arguments",
                    cassette_path=normalized_path,
                    categories=normalized_categories,
                )
            )
            result.extend(
                _scan_payload(
                    tool_call.result,
                    location=f"{call_location}.result",
                    cassette_path=normalized_path,
                    categories=normalized_categories,
                )
            )

    return result


def scan_agent_run(
    run: AgentRun,
    *,
    cassette_path: str | Path | None = None,
    categories: Iterable[str] | str | None = None,
) -> PiiScanResult:
    """Compatibility alias for scan_run."""
    return scan_run(run, cassette_path=cassette_path, categories=categories)


def scan_cassette_file(
    path: str | Path,
    *,
    categories: Iterable[str] | str | None = None,
) -> PiiScanResult:
    """Load and scan one .agentrun.json cassette file."""
    from agentcontract.serialization import load_run

    cassette = Path(path)
    result = scan_run(load_run(cassette), cassette_path=cassette, categories=categories)
    result.scanned_files.append(str(cassette))
    return result


def scan_cassette_path(
    path: str | Path,
    *,
    categories: Iterable[str] | str | None = None,
) -> PiiScanResult:
    """Scan a cassette file or recursively scan a directory of cassette files."""
    target = Path(path)
    if target.is_file():
        return scan_cassette_file(target, categories=categories)
    if not target.exists():
        raise FileNotFoundError(target)
    if not target.is_dir():
        raise ValueError(f"{target} is not a file or directory")

    result = PiiScanResult()
    for cassette in sorted(target.rglob("*.agentrun.json")):
        if cassette.is_file():
            result.extend(scan_cassette_file(cassette, categories=categories))
    return result


def luhn_valid(value: str) -> bool:
    """Return True when the value's digits pass the Luhn checksum."""
    digits = [int(char) for char in value if char.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def _scan_text(
    text: str,
    *,
    location: str,
    cassette_path: str | None,
    categories: set[str],
) -> PiiScanResult:
    result = PiiScanResult()
    if not text or not categories:
        return result

    if "email" in categories:
        for match in _EMAIL_RE.finditer(text):
            result.findings.append(
                PiiFinding(
                    category="email",
                    location=location,
                    snippet=_mask_email(match.group(0)),
                    cassette_path=cassette_path,
                )
            )

    if "phone" in categories:
        for match in _PHONE_RE.finditer(text):
            result.findings.append(
                PiiFinding(
                    category="phone",
                    location=location,
                    snippet=_mask_keep_last_digits(match.group(0), last=4),
                    cassette_path=cassette_path,
                )
            )

    if "ssn" in categories:
        for match in _SSN_RE.finditer(text):
            result.findings.append(
                PiiFinding(
                    category="ssn",
                    location=location,
                    snippet=_mask_keep_last_digits(match.group(0), last=4),
                    cassette_path=cassette_path,
                )
            )

    if "credit_card" in categories:
        for match in _CREDIT_CARD_RE.finditer(text):
            candidate = match.group(0)
            if luhn_valid(candidate):
                result.findings.append(
                    PiiFinding(
                        category="credit_card",
                        location=location,
                        snippet=_mask_keep_last_digits(candidate, last=4),
                        cassette_path=cassette_path,
                    )
                )

    return result


def _scan_payload(
    payload: Any,
    *,
    location: str,
    cassette_path: str | None,
    categories: set[str],
) -> PiiScanResult:
    result = PiiScanResult()

    if isinstance(payload, dict):
        for index, (key, value) in enumerate(payload.items()):
            key_text = str(key)
            entry_location = _dict_entry_location(location, key_text, index)
            result.extend(
                _scan_text(
                    key_text,
                    location=_join_location(entry_location, "__key__"),
                    cassette_path=cassette_path,
                    categories=categories,
                )
            )
            result.extend(
                _scan_payload(
                    value,
                    location=entry_location,
                    cassette_path=cassette_path,
                    categories=categories,
                )
            )
        return result

    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            result.extend(
                _scan_payload(
                    value,
                    location=_join_location(location, index),
                    cassette_path=cassette_path,
                    categories=categories,
                )
            )
        return result

    if payload is None or isinstance(payload, bool):
        return result

    return _scan_text(
        str(payload),
        location=location,
        cassette_path=cassette_path,
        categories=categories,
    )


def _normalize_categories(categories: Iterable[str] | str | None) -> set[str]:
    if categories is None:
        return set(SUPPORTED_CATEGORIES)
    if isinstance(categories, str):
        raw_categories: Iterable[str] = [categories]
    else:
        raw_categories = categories

    normalized = {_normalize_category(category) for category in raw_categories}
    unknown = sorted(normalized.difference(SUPPORTED_CATEGORIES))
    if unknown:
        safe_unknown = ", ".join(_safe_unknown_category(category) for category in unknown)
        raise ValueError(f"Unsupported PII categories: {safe_unknown}")
    return normalized


def _normalize_category(category: str) -> str:
    normalized = str(category).strip().lower()
    return _CATEGORY_ALIASES.get(normalized, normalized)


def _join_location(parent: str, child: str | int) -> str:
    if isinstance(child, int):
        suffix = f"[{child}]"
        return f"{parent}{suffix}" if parent else suffix

    if _LOCATION_IDENTIFIER_RE.match(child):
        return f"{parent}.{child}" if parent else child

    suffix = f"[{json.dumps(child)}]"
    return f"{parent}{suffix}" if parent else suffix


def _dict_entry_location(parent: str, key: str, index: int) -> str:
    if _text_has_pii(key):
        suffix = f"[<key:{index}>]"
        return f"{parent}{suffix}" if parent else suffix
    return _join_location(parent, key)


def _text_has_pii(text: str) -> bool:
    return _scan_text(
        text,
        location="",
        cassette_path=None,
        categories=set(SUPPORTED_CATEGORIES),
    ).has_findings


def _safe_unknown_category(category: str) -> str:
    if _text_has_pii(category):
        return "<redacted>"
    return category


def _mask_email(value: str) -> str:
    local, separator, domain = value.partition("@")
    if not separator:
        return "***"

    labels = domain.split(".")
    if labels:
        labels[0] = _mask_token(labels[0])
    return f"{_mask_token(local)}@{'.'.join(labels)}"


def _mask_token(value: str) -> str:
    if not value:
        return "***"
    return f"{value[0]}***"


def _mask_keep_last_digits(value: str, *, last: int) -> str:
    total_digits = sum(1 for char in value if char.isdigit())
    mask_until = max(total_digits - last, 0)
    seen_digits = 0
    masked = []

    for char in value:
        if not char.isdigit():
            masked.append(char)
            continue

        seen_digits += 1
        masked.append("*" if seen_digits <= mask_until else char)

    return "".join(masked)


def _path_to_str(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


# Backwards-compatible public aliases for callers that prefer acronym casing.
PIIFinding = PiiFinding
PIIScanResult = PiiScanResult
