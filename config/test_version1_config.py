# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Version 1 hybrid config: GPT-5.6 Terra for Draft/Sketch, BFS-Prover via vLLM."""

import os

from src.paths import (
    BABEL_FORMAL_SINGLE_DATASET,
    BFS_PROVER_SERVED_MODEL,
    MATHLIB_DIR,
    PROJECT_ROOT,
    VLLM_BASE_URL,
)


# Use test_version1.py so DSP+ installs generator_version1.py before launching.

data = str(BABEL_FORMAL_SINGLE_DATASET)
split = ["test"]
target_dir = str(PROJECT_ROOT / "result" / "test_version1_babel_formal_v1_single")

concurrent_num = 1
attempts = 1

draft_max_running_requests = 1
sketch_max_running_requests = 1

sketch_leanserver_num = 1
prove_leanserver_num = 2

openai_client = [
    {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ["OPENAI_API_KEY"],
    }
]

draft_model_config = openai_client
draft_sample_config = {
    "model": "gpt-5.6-terra",
    "reasoning_effort": "medium",
    "timeout": 3600,
    "max_tokens": 32768,
}

sketch_model_config = openai_client
sketch_sample_config = {
    "model": "gpt-5.6-terra",
    "reasoning_effort": "high",
    "timeout": 3600,
    "max_tokens": 32768,
}

sketch_verify_config = {
    "verify_timeout": 180,
    "cwd": str(MATHLIB_DIR),
}

prove_model_config = [
    {
        "base_url": VLLM_BASE_URL,
        "api_key": "EMPTY",
    }
]

prove_sampling_config = {
    "name_lean_copilot": "BFS-Prover-API",
    "model": BFS_PROVER_SERVED_MODEL,
    "temperature": 1.1,
    "top_p": 1,
    "timeout": 1800,
    "max_tokens": 64,
    "n": 8,
    "max_output": 4,
    "use_beam_search": False,
}

prove_verify_config = {
    "verify_timeout": 1200,
    "max_tree_size": 64,
    "search_attempts": 4,
    "port_lean_copilot": 23338,
    "cwd": str(MATHLIB_DIR),
}

