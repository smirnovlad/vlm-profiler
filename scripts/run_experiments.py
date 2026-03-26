#!/usr/bin/env python3
"""CLI entry point for running VLM profiling experiments."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner import run_experiments
from src.utils import kill_my_gpu_processes, setup_logging


def build_output_dir(base: str, exp_name: str) -> Path:
    """Build output path: {base}/{date}/{exp_name}/{time}."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    return Path(base) / date_str / exp_name / time_str


def setup_file_logging(output_dir: Path):
    """Add a file handler to the root logger."""
    log_path = output_dir / "experiment.log"
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger(__name__).info("Logging to %s", log_path)


def main():
    parser = argparse.ArgumentParser(description="Run VLM profiling experiments")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--output-base",
        default="outputs",
        help="Base directory for outputs (default: outputs)",
    )
    parser.add_argument(
        "--exp-name",
        default="default",
        help="Experiment name (used in output path)",
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
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Wandb project name (enables wandb logging)",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Wandb entity (team/org)",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Wandb run name",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU device index (default: 0)",
    )
    args = parser.parse_args()

    # Build structured output directory
    output_dir = build_output_dir(args.output_base, args.exp_name)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    setup_file_logging(output_dir)

    logger = logging.getLogger(__name__)
    logger.info("Output directory: %s", output_dir)

    if args.free_gpus:
        kill_my_gpu_processes()

    run_experiments(
        config_path=args.config,
        results_dir=str(results_dir),
        models_filter=args.models,
        datasets_filter=args.datasets,
        devices_filter=args.devices,
        gpu_index=args.gpu_index,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    logger.info("All results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
