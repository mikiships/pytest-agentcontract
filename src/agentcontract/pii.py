"""Local, read-only heuristics for potential cassette PII exposure."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Iterator
from pathlib import Path

PATTERNS = {
    "email": re.compile(r"(?<![\w.+%-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)"),
    "us_phone": re.compile(
        r"(?<![\w+])(?:\+?1[ .-]?)?"
        r"(?:\([2-9][0-9]{2}\)[ ]?|[2-9][0-9]{2}[ .-])"
        r"[2-9][0-9]{2}[ .-][0-9]{4}(?!\w)"
    ),
    "us_ssn": re.compile(r"(?<![\w-])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![\w-])"),
}


class _Object(list[tuple[str, object]]):
    """Preserve all JSON object entries, including duplicate keys."""


def scan_value(value: object) -> Iterator[tuple[str, str]]:
    """Yield safe ordinal locations and categories, never source text."""
    pending = [("$", value)]
    while pending:
        location, item = pending.pop()
        if isinstance(item, str):
            for category, pattern in PATTERNS.items():
                if pattern.search(item):
                    yield location, category
        elif isinstance(item, (_Object, dict)):
            entries = list(item.items()) if isinstance(item, dict) else item
            for index in reversed(range(len(entries))):
                key, child = entries[index]
                pending.append((f"{location}.entry[{index}].value", child))
                pending.append((f"{location}.entry[{index}].key", key))
        elif isinstance(item, list):
            for index in reversed(range(len(item))):
                pending.append((f"{location}[{index}]", item[index]))


def _invalid_constant(_value: str) -> object:
    raise ValueError("invalid JSON constant")


def scan_path(target: Path) -> dict[str, object]:
    """Scan raw JSON; return only sanitized findings, errors and counts."""
    paths: list[Path] = []
    errors: list[dict[str, str]] = []

    def discovery_error(_error: OSError | None = None) -> None:
        errors.append({"file": "target", "category": "discovery_failed"})

    try:
        mode = target.stat().st_mode
        if stat.S_ISREG(mode):
            paths = [target]
        elif stat.S_ISDIR(mode):
            for root, dirs, files in os.walk(target, onerror=discovery_error):
                # Do not silently omit linked directories or traverse outside the tree.
                for name in dirs[:]:
                    if (Path(root) / name).is_symlink():
                        discovery_error()
                        dirs.remove(name)
                paths.extend(Path(root) / name for name in files if name.endswith(".agentrun.json"))
        else:
            discovery_error()
    except (OSError, ValueError):
        discovery_error()

    if not paths and not errors:
        errors.append({"file": "target", "category": "empty_target"})
    findings: list[dict[str, str]] = []
    scanned = 0
    for index, path in enumerate(sorted(paths)):
        identifier = f"file-{index + 1:04d}"
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("not a regular file")
            payload = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_Object,
                parse_constant=_invalid_constant,
            )
            for location, category in scan_value(payload):
                findings.append({"file": identifier, "location": location, "category": category})
            scanned += 1
        except (OSError, ValueError, RecursionError):
            errors.append({"file": identifier, "category": "unscannable_file"})
    return {"scanned_files": scanned, "findings": findings, "errors": errors}
