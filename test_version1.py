#!/usr/bin/env python3
"""Startup script for the version 1 DSP+ hybrid experiment."""

from __future__ import annotations

import argparse
import os

from src.dsp_runner import preflight, run_dsp_workflow
from src.paths import PROJECT_ROOT, TEST_VERSION1_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DSP+ version 1 with GPT-5.6 Terra and local BFS-Prover.")
    parser.add_argument("--config", default=os.fspath(TEST_VERSION1_CONFIG), help="DSP+ config path.")
    parser.add_argument("--node_rank", "--node-rank", dest="node_rank", type=int, default=0)
    parser.add_argument("--world_size", "--world-size", dest="world_size", type=int, default=1)
    parser.add_argument("--skip-vllm-check", action="store_true", help="Skip the local BFS-Prover server check.")
    parser.add_argument("--preflight-only", action="store_true", help="Check configuration and exit without running DSP+.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Config: {args.config}", flush=True)

    preflight(args.config, check_vllm=not args.skip_vllm_check)
    if args.preflight_only:
        print("Preflight passed.", flush=True)
        return

    run_dsp_workflow(
        config_path=args.config,
        node_rank=args.node_rank,
        world_size=args.world_size,
    )


if __name__ == "__main__":
    main()
