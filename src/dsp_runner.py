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
LAKE_PATH = ELAN_BIN_DIR / "lake"


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
    if shutil.which("elan") is None:
        errors.append(
            "elan is not on PATH. Install Lean/Elan or add ~/.elan/bin to PATH: "
            'export PATH="$HOME/.elan/bin:$PATH"'
        )
    if not LAKE_PATH.exists():
        errors.append(
            f"lake was not found at {LAKE_PATH}. DSP+'s verifier starts Lean with this path."
        )
    elif not os.access(LAKE_PATH, os.X_OK):
        errors.append(f"lake exists but is not executable: {LAKE_PATH}")
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
