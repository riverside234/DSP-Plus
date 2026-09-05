"""Central paths for the version 1 DSP+ experiment on the lab machine."""

from pathlib import PurePosixPath


PROJECT_ROOT = PurePosixPath("/data/not_backed_up/yxu209/DSP-Plus")

CONFIG_DIR = PROJECT_ROOT / "config"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULT_DIR = PROJECT_ROOT / "result"
SRC_DIR = PROJECT_ROOT / "src"
VLLM_DIR = PROJECT_ROOT / "vllm"
VLLM_MODELS_DIR = VLLM_DIR / "models"
MATHLIB_DIR = PROJECT_ROOT / "mathlib4"

DSP_WORKFLOW = PROJECT_ROOT / "dsp_workflow.py"
TEST_VERSION1_CONFIG = CONFIG_DIR / "test_version1_config.py"
ENV_FILE = PROJECT_ROOT / ".env"

MINIF2F_DATASET = DATASETS_DIR / "minif2f.jsonl"
BABEL_FORMAL_SINGLE_DATASET = DATASETS_DIR / "babel_formal_v1_single_dsp.jsonl"
BABEL_FORMAL_FULL_DATASET = DATASETS_DIR / "babel_formal_v1_dsp.jsonl"

BFS_PROVER_MODEL_DIR = VLLM_MODELS_DIR / "BFS-Prover-V1-7B"
BFS_PROVER_HF_MODEL = "ByteDance-Seed/BFS-Prover-V1-7B"
BFS_PROVER_REVISION = "610fef2d1fc714ea70adb8f3e089eb35c829d2b0"
BFS_PROVER_SERVED_MODEL = "bytedance-research/BFS-Prover"
VLLM_BASE_URL = "http://127.0.0.1:30001/v1"

LEAN_TOOLCHAIN = "leanprover/lean4:v4.17.0-rc1"


def project_path(*parts: str) -> PurePosixPath:
    """Build an absolute path under PROJECT_ROOT."""
    return PROJECT_ROOT.joinpath(*parts)
