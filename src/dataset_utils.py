"""Dataset helpers for version 1 DSP+ runs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from src.jsonl_io import PathLike, iter_jsonl, resolve_project_path, write_jsonl
from src.paths import DATASETS_DIR, MINIF2F_DATASET


REQUIRED_DSP_FIELDS = ("name", "split", "header", "formal_statement")


@dataclass(frozen=True)
class DatasetSubset:
    path: Path
    count: int
    names: list[str]


def dataset_path(filename: str) -> PurePosixPath:
    """Return an absolute dataset path under the DSP+ datasets directory."""
    return DATASETS_DIR / filename


def records_for_split(records: Iterable[dict], split: str | Iterable[str]) -> list[dict]:
    """Filter records by DSP+ split name."""
    split_names = {split} if isinstance(split, str) else set(split)
    return [record for record in records if record.get("split") in split_names]


def validate_dsp_records(records: Iterable[dict], required_fields: Iterable[str] = REQUIRED_DSP_FIELDS) -> list[str]:
    """Return validation errors for records missing DSP+ fields."""
    errors: list[str] = []
    required = tuple(required_fields)
    for index, record in enumerate(records, start=1):
        name = record.get("name", f"record_{index}")
        for field in required:
            if field not in record or record[field] in (None, ""):
                errors.append(f"{name}: missing required field '{field}'")
    return errors


def ensure_dataset_exists(path: PathLike) -> Path:
    """Resolve a dataset path and raise a clear error if it is missing."""
    dataset = resolve_project_path(path)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    return dataset


def make_minif2f_subset(n: int, seed: int = 42, split: str = "test", source: PathLike = MINIF2F_DATASET, target: PathLike | None = None) -> DatasetSubset:
    """Create a deterministic miniF2F subset under datasets/."""
    if n <= 0:
        raise ValueError("n must be positive")

    source_path = ensure_dataset_exists(source)
    rows = records_for_split(iter_jsonl(source_path), split)
    selected = random.Random(seed).sample(rows, min(n, len(rows)))

    target_path = target or dataset_path(f"minif2f_{split}_{n}.jsonl")
    written_path = write_jsonl(selected, target_path)
    return DatasetSubset(
        path=written_path,
        count=len(selected),
        names=[str(record.get("name", "")) for record in selected],
    )


def validate_dsp_jsonl(path: PathLike) -> list[str]:
    """Validate the required DSP+ fields in a JSONL dataset."""
    return validate_dsp_records(iter_jsonl(path))

