"""Result helpers for DSP+ experiment directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.jsonl_io import PathLike, resolve_project_path


@dataclass(frozen=True)
class ResultSummary:
    target_dir: Path
    problem_dirs: int
    finished_count: int
    finished_files: list[Path]


def iter_problem_dirs(target_dir: PathLike) -> list[Path]:
    """Return immediate problem result directories under a DSP+ target dir."""
    target = resolve_project_path(target_dir)
    if not target.exists():
        return []
    return sorted(path for path in target.iterdir() if path.is_dir())


def find_finished_files(target_dir: PathLike) -> list[Path]:
    """Find finished.txt files under a DSP+ target dir."""
    target = resolve_project_path(target_dir)
    if not target.exists():
        return []
    return sorted(target.rglob("finished.txt"))


def summarize_results(target_dir: PathLike) -> ResultSummary:
    """Summarize completed DSP+ problem directories."""
    target = resolve_project_path(target_dir)
    finished = find_finished_files(target)
    return ResultSummary(
        target_dir=target,
        problem_dirs=len(iter_problem_dirs(target)),
        finished_count=len(finished),
        finished_files=finished,
    )


def format_result_summary(summary: ResultSummary) -> str:
    """Format a compact result summary for logs."""
    return (
        f"target_dir={summary.target_dir}\n"
        f"problem_dirs={summary.problem_dirs}\n"
        f"finished_count={summary.finished_count}"
    )

