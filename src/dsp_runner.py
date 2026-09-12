"""Runtime helpers for the version 1 DSP+ startup script."""

from __future__ import annotations

import os
import runpy
import shutil
import sys
import urllib.request
from pathlib import Path

from src.dataset_utils import ensure_dataset_exists
from src.env import load_project_env
from src.jsonl_io import resolve_project_path
from src.paths import DSP_WORKFLOW, MATHLIB_DIR, PROJECT_ROOT, TEST_VERSION1_CONFIG, VLLM_BASE_URL


ELAN_BIN_DIR = Path.home() / ".elan" / "bin"


def _is_executable(path: str | os.PathLike[str]) -> bool:
    return os.path.exists(path) and os.access(path, os.X_OK)


def _resolve_tool(tool_name: str, env_name: str) -> str | None:
    configured = os.environ.get(env_name)
    if configured:
        return configured

    configured_bin = os.environ.get("LEAN_TOOLCHAIN_BIN")
    if configured_bin:
        candidate = Path(configured_bin) / tool_name
        if candidate.exists():
            return os.fspath(candidate)

    path_tool = shutil.which(tool_name)
    if path_tool:
        return path_tool

    fallback = ELAN_BIN_DIR / tool_name
    if fallback.exists():
        return os.fspath(fallback)

    return None


def resolve_elan_path() -> str | None:
    return _resolve_tool("elan", "DSP_ELAN_PATH")


def resolve_lake_path() -> str | None:
    return _resolve_tool("lake", "DSP_LAKE_PATH")


def install_version1_generator() -> None:
    """Route Draft and Sketch LLM scheduling through generator_version1.py."""
    import dsp.draft as draft_module
    import dsp.sketch as sketch_module
    import dsp.worker as worker_module
    from dsp.worker.generator_version1 import LLMServerScheduler

    worker_module.LLMServerScheduler = LLMServerScheduler
    draft_module.LLMServerScheduler = LLMServerScheduler
    sketch_module.LLMServerScheduler = LLMServerScheduler


def check_bfs_server(base_url: str = VLLM_BASE_URL, timeout: float = 5.0) -> None:
    """Raise if the local BFS-Prover vLLM server is not reachable."""
    models_url = f"{base_url.rstrip('/')}/models"
    with urllib.request.urlopen(models_url, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(f"BFS-Prover server returned HTTP {response.status}: {models_url}")


def check_lean_toolchain() -> list[str]:
    """Return actionable errors for missing Lean/Elan tools."""
    errors: list[str] = []
    elan_path = resolve_elan_path()
    lake_path = resolve_lake_path()

    if elan_path is None:
        errors.append(
            "elan was not found. Add it to PATH, set LEAN_TOOLCHAIN_BIN, or set DSP_ELAN_PATH."
        )
    elif not _is_executable(elan_path):
        errors.append(f"elan exists but is not executable: {elan_path}")

    if lake_path is None:
        errors.append(
            "lake was not found. Add it to PATH, set LEAN_TOOLCHAIN_BIN, or set DSP_LAKE_PATH."
        )
    elif not _is_executable(lake_path):
        errors.append(f"lake exists but is not executable: {lake_path}")
    return errors


def load_config(config_path: str | os.PathLike[str]):
    """Load a DSP+ config through the repository loader."""
    from dsp.utils import load_config as dsp_load_config

    load_project_env()
    return dsp_load_config(os.fspath(config_path))


def preflight(config_path: str | os.PathLike[str] = TEST_VERSION1_CONFIG, check_vllm: bool = True) -> None:
    """Check the lab-root files and services needed by test_version1.py."""
    errors: list[str] = []
    project_root = Path(os.fspath(PROJECT_ROOT))
    workflow_path = Path(os.fspath(DSP_WORKFLOW))
    mathlib_path = Path(os.fspath(MATHLIB_DIR))
    config_file = resolve_project_path(config_path)
    env_file = load_project_env()

    if not project_root.exists():
        errors.append(f"Project root does not exist: {project_root}")
    if not workflow_path.exists():
        errors.append(f"DSP workflow script does not exist: {workflow_path}")
    if not mathlib_path.exists():
        errors.append(f"mathlib4 directory does not exist: {mathlib_path}")
    if not config_file.exists():
        errors.append(f"Config file does not exist: {config_file}")
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append(f"OPENAI_API_KEY is not set; add it to {env_file or PROJECT_ROOT / '.env'}")
    errors.extend(check_lean_toolchain())

    cfg = None
    if not errors:
        try:
            cfg = load_config(config_file)
        except Exception as exc:
            errors.append(f"Failed to load config {config_file}: {exc}")

    if cfg is not None:
        try:
            ensure_dataset_exists(cfg.data)
        except Exception as exc:
            errors.append(str(exc))

    if check_vllm:
        try:
            check_bfs_server()
        except Exception as exc:
            errors.append(f"BFS-Prover vLLM server is not ready: {exc}")

    if errors:
        message = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"Preflight failed:\n{message}")


def run_dsp_workflow(
    config_path: str | os.PathLike[str] = TEST_VERSION1_CONFIG,
    node_rank: int = 0,
    world_size: int = 1,
) -> None:
    """Run dsp_workflow.py in-process after installing version 1 LLM scheduling."""
    load_project_env()
    lake_path = resolve_lake_path()
    if lake_path is not None:
        os.environ.setdefault("DSP_LAKE_PATH", lake_path)
    install_version1_generator()

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    try:
        os.chdir(os.fspath(PROJECT_ROOT))
        sys.argv = [
            os.fspath(DSP_WORKFLOW),
            "--config",
            os.fspath(resolve_project_path(config_path)),
            "--node_rank",
            str(node_rank),
            "--world_size",
            str(world_size),
        ]
        runpy.run_path(os.fspath(DSP_WORKFLOW), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
