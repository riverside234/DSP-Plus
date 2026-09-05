"""JSONL helpers for DSP+ datasets and result metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.paths import PROJECT_ROOT


PathLike = str | os.PathLike[str]


def resolve_project_path(path: PathLike) -> Path:
    """Resolve relative paths under the lab DSP+ project root."""
    raw_path = os.fspath(path)
    if raw_path.startswith("/"):
        return Path(raw_path)
    return Path(os.fspath(PROJECT_ROOT / raw_path))


def iter_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    jsonl_path = resolve_project_path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{jsonl_path}:{line_number}: expected a JSON object")
            yield value


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    return list(iter_jsonl(path))


def write_jsonl(records: Iterable[dict[str, Any]], path: PathLike) -> Path:
    """Write records to a JSONL file and return the resolved path."""
    jsonl_path = resolve_project_path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl_path


def count_jsonl(path: PathLike) -> int:
    """Count non-empty JSONL records."""
    return sum(1 for _ in iter_jsonl(path))

