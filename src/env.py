"""Environment loading for local DSP+ experiment secrets."""

from __future__ import annotations

import os
from pathlib import Path

from src.paths import ENV_FILE


def _strip_optional_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env(env_path: str | os.PathLike[str] = ENV_FILE, override: bool = False) -> Path | None:
    """Load KEY=VALUE entries from the project .env file into os.environ."""
    path = Path(os.fspath(env_path))
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if not override and key in os.environ:
                continue
            os.environ[key] = _strip_optional_quotes(value)

    return path

