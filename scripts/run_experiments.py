#!/usr/bin/env python3
"""CLI entry point for running VLM profiling experiments."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import run_experiments
from src.utils import kill_my_gpu_processes, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run VLM profiling experiments")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to store JSON results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter: only run these models",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Filter: only run these datasets (scienceqa, textvqa, coco_caption)",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Filter: only run on these devices (cuda, cpu)",
    )
    parser.add_argument(
        "--free-gpus",
        action="store_true",
        help="Kill own GPU processes before starting",
    )
    args = parser.parse_args()

    setup_logging()

    if args.free_gpus:
        kill_my_gpu_processes()

    run_experiments(
        config_path=args.config,
        results_dir=args.results_dir,
        models_filter=args.models,
        datasets_filter=args.datasets,
        devices_filter=args.devices,
    )


if __name__ == "__main__":
    main()
