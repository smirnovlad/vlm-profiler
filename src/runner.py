"""Main experiment runner: iterates over the config matrix and collects results."""

import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.loader import VLMSample, load_dataset_by_name
from src.data.preprocessing import generate_prompt, resize_image
from src.models.registry import LoadedModel, load_model
from src.profiling.energy import EnergyMonitor
from src.profiling.latency import measure_latency
from src.profiling.quality import compute_quality
from src.utils import clear_gpu_memory, is_valid_combo

logger = logging.getLogger(__name__)


def _make_experiment_id(
    model_name: str,
    dataset_name: str,
    resolution: int,
    prompt_length: int,
    device: str,
    optimization: str,
    batch_size: int,
) -> str:
    safe_model = model_name.replace("/", "_")
    return (
        f"{safe_model}__ds={dataset_name}__res={resolution}__plen={prompt_length}"
        f"__dev={device}__opt={optimization}__bs={batch_size}"
    )


def _prepare_batch(
    loaded: LoadedModel,
    samples: list[VLMSample],
    resolution: int,
    prompt_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Prepare a batch of inputs for the model."""
    batch_samples = samples[:batch_size]
    images = [resize_image(s.image, resolution) for s in batch_samples]
    prompts = [generate_prompt(s.question, prompt_length) for s in batch_samples]

    if batch_size == 1:
        inputs = loaded.processor(
            images=images[0], text=prompts[0], return_tensors="pt"
        )
    else:
        inputs = loaded.processor(
            images=images, text=prompts, return_tensors="pt", padding=True
        )

    device = loaded.device
    if device == "cuda":
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


def run_single_experiment(
    loaded: LoadedModel,
    samples: list[VLMSample],
    resolution: int,
    prompt_length: int,
    batch_size: int,
    warmup_runs: int = 3,
    timed_runs: int = 10,
    gpu_index: int = 0,
) -> dict[str, Any]:
    """Run a single experiment configuration and return results dict."""
    exp_id = _make_experiment_id(
        loaded.model_name,
        samples[0].dataset_name,
        resolution,
        prompt_length,
        loaded.device,
        loaded.optimization,
        batch_size,
    )
    logger.info("Running experiment: %s", exp_id)

    result: dict[str, Any] = {
        "experiment_id": exp_id,
        "model": loaded.model_name,
        "dataset": samples[0].dataset_name,
        "resolution": resolution,
        "prompt_length": prompt_length,
        "device": loaded.device,
        "optimization": loaded.optimization,
        "batch_size": batch_size,
    }

    try:
        inputs = _prepare_batch(loaded, samples, resolution, prompt_length, batch_size)

        # Latency
        def inference_fn():
            with torch.no_grad():
                loaded.model.generate(**inputs, max_new_tokens=50)

        latency = measure_latency(
            inference_fn,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            device=loaded.device,
        )
        result["latency"] = asdict(latency)

        # Energy
        if loaded.device == "cuda":
            monitor = EnergyMonitor(gpu_index=gpu_index)
            monitor.start()
            for _ in range(timed_runs):
                with torch.no_grad():
                    loaded.model.generate(**inputs, max_new_tokens=50)
                torch.cuda.synchronize()
            energy = monitor.stop()
            result["energy"] = asdict(energy)
        else:
            result["energy"] = None

        # Quality (run on all samples)
        predictions = []
        references = []
        for sample in samples:
            img = resize_image(sample.image, resolution)
            prompt = generate_prompt(sample.question, prompt_length)
            inp = loaded.processor(images=img, text=prompt, return_tensors="pt")
            if loaded.device == "cuda":
                inp = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inp.items()}
            with torch.no_grad():
                output_ids = loaded.model.generate(**inp, max_new_tokens=50)
            decoded = loaded.processor.batch_decode(output_ids, skip_special_tokens=True)
            predictions.append(decoded[0] if decoded else "")
            references.append(sample.answer)

        quality = compute_quality(predictions, references)
        result["quality"] = asdict(quality)
        result["status"] = "success"

    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM for experiment %s", exp_id)
        result["status"] = "oom"
        clear_gpu_memory()
    except Exception as e:
        logger.error("Error in experiment %s: %s", exp_id, e)
        result["status"] = "error"
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_completed_experiments(results_dir: Path) -> set[str]:
    """Scan results dir for already-completed experiment IDs."""
    completed = set()
    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                if data.get("status") == "success":
                    completed.add(data["experiment_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def save_result(result: dict, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_id = result["experiment_id"]
    path = results_dir / f"{exp_id}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Saved result: %s", path)


def run_experiments(
    config_path: str,
    results_dir: str = "results",
    models_filter: list[str] | None = None,
    datasets_filter: list[str] | None = None,
    devices_filter: list[str] | None = None,
):
    """Run the full experiment matrix from config."""
    config = load_config(config_path)
    results_path = Path(results_dir)
    completed = get_completed_experiments(results_path)

    # Load datasets
    dataset_cache: dict[str, list[VLMSample]] = {}
    for ds_cfg in config["datasets"]:
        ds_name = ds_cfg["name"]
        if datasets_filter and ds_name not in datasets_filter:
            continue
        dataset_cache[ds_name] = load_dataset_by_name(ds_name, ds_cfg["num_samples"])

    scaling = config["scaling"]

    for model_name in config["models"]:
        if models_filter and model_name not in models_filter:
            continue

        for optimization in scaling["optimizations"]:
            for device in scaling["devices"]:
                if devices_filter and device not in devices_filter:
                    continue
                if not is_valid_combo(device, optimization):
                    continue

                # Load model once per (model, optimization, device) combo
                try:
                    loaded = load_model(model_name, device, optimization)
                except Exception as e:
                    logger.error(
                        "Failed to load %s (opt=%s, dev=%s): %s",
                        model_name, optimization, device, e,
                    )
                    continue

                for ds_name, samples in dataset_cache.items():
                    for resolution in scaling["resolutions"]:
                        for prompt_length in scaling["prompt_lengths"]:
                            for batch_size in scaling["batch_sizes"]:
                                exp_id = _make_experiment_id(
                                    model_name, ds_name, resolution,
                                    prompt_length, device, optimization, batch_size,
                                )
                                if exp_id in completed:
                                    logger.info("Skipping (done): %s", exp_id)
                                    continue

                                result = run_single_experiment(
                                    loaded, samples, resolution, prompt_length,
                                    batch_size,
                                    warmup_runs=config["profiling"]["warmup_runs"],
                                    timed_runs=config["profiling"]["timed_runs"],
                                )
                                save_result(result, results_path)

                # Free model memory before loading next
                del loaded
                clear_gpu_memory()
