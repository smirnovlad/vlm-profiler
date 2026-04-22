#!/usr/bin/env python3
"""Measure per-component latency and prefill/decode split for each model.

Runs at baseline config (res=224, plen=10, bs=1, CUDA, FP32) on one dataset
sample. Writes results to a single JSON at results/component_breakdown.json,
which generate_report.py picks up for the component breakdown chart.

Usage:
    python scripts/component_breakdown.py --models Salesforce/blip2-opt-2.7b \
        --gpu-index 0

    # Or all configured models:
    python scripts/component_breakdown.py --gpu-index 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_dataset_by_name  # noqa: E402
from src.data.preprocessing import format_prompt_for_model, resize_image  # noqa: E402
from src.models.registry import load_model  # noqa: E402
from src.profiling.components import (  # noqa: E402
    measure_component_latency,
    measure_prefill_decode,
)
from src.utils import clear_gpu_memory, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

BASELINE = {
    "resolution": 224,
    "prompt_length": 10,
    "batch_size": 1,
    "dataset": "scienceqa",
}

OUTPUT_PATH = Path("results/component_breakdown.json")


def _build_inputs(loaded, sample, resolution: int):
    image = resize_image(sample.image, resolution)
    prompt = format_prompt_for_model(sample.question, loaded.model_name)
    inputs = loaded.processor(images=image, text=prompt, return_tensors="pt")
    if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "dim"):
        pv = inputs["pixel_values"]
        if pv.dim() == 5 and pv.shape[1] == 1:
            inputs["pixel_values"] = pv.squeeze(1)
    device_str = f"cuda:{loaded.gpu_index}"
    return {k: v.to(device_str) if hasattr(v, "to") else v for k, v in inputs.items()}


def _make_generate_fn(loaded):
    def gen(inputs, max_new_tokens: int = 50):
        return loaded.model.generate(**inputs, max_new_tokens=max_new_tokens)
    return gen


def measure_one_model(
    model_name: str,
    sample,
    gpu_index: int,
    warmup_runs: int = 2,
    timed_runs: int = 3,
    optimization: str = "none",
) -> dict:
    logger.info(
        "=== %s (opt=%s, warmup=%d, timed=%d) ===",
        model_name, optimization, warmup_runs, timed_runs,
    )
    loaded = load_model(
        model_name, device="cuda", optimization=optimization, gpu_index=gpu_index
    )
    try:
        inputs = _build_inputs(loaded, sample, BASELINE["resolution"])
        gen_fn = _make_generate_fn(loaded)

        comp = measure_component_latency(
            loaded.model,
            inputs,
            generate_fn=lambda inp: gen_fn(inp, max_new_tokens=50),
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )
        pd = measure_prefill_decode(
            gen_fn,
            inputs,
            short_n=1,
            long_n=50,
            warmup_runs=max(warmup_runs - 1, 1),
            timed_runs=timed_runs,
        )

        record = {
            "model": model_name,
            "baseline": BASELINE,
            "component_times": asdict(comp),
            "prefill_decode": asdict(pd),
        }
        logger.info(
            "wall=%.1fms, measured=%.1fms, prefill=%.1fms, decode/token=%.2fms",
            comp.wall_ms, comp.total_measured_ms,
            pd.prefill_ms, pd.decode_per_token_ms,
        )
        for name, ms in sorted(comp.per_call_ms.items(), key=lambda kv: -kv[1]):
            logger.info("  %-26s %7.2f ms  (%5.1f%%)",
                        name, ms, 100.0 * comp.per_call_share[name])
        return record
    finally:
        del loaded
        clear_gpu_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
        help="Config file listing models",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Override model list (default: all from config)",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument(
        "--optimization", default="none",
        choices=["none", "fp16", "flash_attn2"],
        help="Load dtype/attention mode (default: none = fp32 baseline)",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help="Where to write the JSON result",
    )
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    models = args.models or cfg["models"]

    # Load one sample (ScienceQA is small and reliable)
    samples = load_dataset_by_name(BASELINE["dataset"], num_samples=1)
    if not samples:
        raise RuntimeError(f"No samples loaded from {BASELINE['dataset']}")
    sample = samples[0]

    results = []
    output_path = Path(args.output)
    # If there's an existing file, preserve previously-measured models so a
    # failure halfway doesn't wipe prior progress.
    existing: dict[str, dict] = {}
    if output_path.exists():
        try:
            prev = json.loads(output_path.read_text())
            for r in prev.get("results", []):
                existing[r["model"]] = r
        except (json.JSONDecodeError, KeyError):
            logger.warning("Existing breakdown file unreadable, starting fresh")

    # When running with a non-baseline optimization, we key records by
    # "{model}@{opt}" so they don't overwrite the fp32 baseline entries.
    key_suffix = "" if args.optimization == "none" else f"@{args.optimization}"

    for model_name in models:
        try:
            rec = measure_one_model(
                model_name,
                sample,
                args.gpu_index,
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
                optimization=args.optimization,
            )
            if key_suffix:
                rec["optimization"] = args.optimization
            existing[model_name + key_suffix] = rec
        except Exception as e:  # pragma: no cover — surfaced via log
            logger.error("Failed on %s: %s", model_name, e)
            existing[model_name + key_suffix] = {
                "model": model_name,
                "baseline": BASELINE,
                "optimization": args.optimization,
                "error": str(e),
            }
        # Persist after every model so an OOM late in the run doesn't lose data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {"results": list(existing.values())},
                indent=2,
                default=str,
            )
        )
        logger.info("Saved interim results to %s", output_path)

    results = list(existing.values())
    logger.info("Done. %d models measured.", len(results))


if __name__ == "__main__":
    main()
