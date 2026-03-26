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

# Wandb handle — initialized lazily via init_wandb()
_wandb_run = None


def init_wandb(
    project: str = "vlm-profiler",
    entity: str | None = None,
    config: dict | None = None,
    run_name: str | None = None,
):
    """Initialize a wandb run for logging experiment results."""
    global _wandb_run
    import wandb

    _wandb_run = wandb.init(
        project=project,
        entity=entity,
        config=config or {},
        name=run_name,
        reinit=True,
    )
    logger.info("Wandb run initialized: %s", _wandb_run.url)
    return _wandb_run


def _log_to_wandb(result: dict[str, Any]):
    """Log a single experiment result to wandb."""
    if _wandb_run is None:
        return

    log_data = {
        "model": result["model"],
        "dataset": result["dataset"],
        "resolution": result["resolution"],
        "prompt_length": result["prompt_length"],
        "device": result["device"],
        "optimization": result["optimization"],
        "batch_size": result["batch_size"],
        "status": result["status"],
    }

    if result.get("status") == "success":
        if result.get("latency"):
            log_data["latency/mean_ms"] = result["latency"]["mean_ms"]
            log_data["latency/std_ms"] = result["latency"]["std_ms"]
            log_data["latency/p50_ms"] = result["latency"]["p50_ms"]
            log_data["latency/p95_ms"] = result["latency"]["p95_ms"]
            log_data["latency/p99_ms"] = result["latency"]["p99_ms"]
        if result.get("energy"):
            log_data["energy/avg_power_w"] = result["energy"]["avg_power_w"]
            log_data["energy/max_power_w"] = result["energy"]["max_power_w"]
            log_data["energy/energy_j"] = result["energy"]["energy_j"]
            log_data["energy/duration_s"] = result["energy"]["duration_s"]
        if result.get("quality"):
            log_data["quality/wer"] = result["quality"]["wer_score"]
            log_data["quality/exact_match"] = result["quality"]["exact_match_accuracy"]
            log_data["quality/num_samples"] = result["quality"]["num_samples"]

    _wandb_run.log(log_data)


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


# Models with non-standard generate() that need custom inference
CUSTOM_INFERENCE_MODELS = {
    "vikhyatk/moondream2",
}


def _is_custom_inference(model_name: str) -> bool:
    return model_name in CUSTOM_INFERENCE_MODELS


def _set_pad_token(processor):
    """Ensure processor's tokenizer has a pad token for batched inference."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


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
        _set_pad_token(loaded.processor)
        inputs = loaded.processor(
            images=images, text=prompts, return_tensors="pt", padding=True
        )

    device = loaded.device
    if device == "cuda":
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


def _run_generate(
    loaded: LoadedModel,
    inputs: dict[str, Any],
    max_new_tokens: int = 50,
    image: "Image.Image | None" = None,
    prompt_text: str | None = None,
):
    """Run model.generate() with model-specific handling."""
    if _is_custom_inference(loaded.model_name) and image is not None:
        # Moondream2 expects PIL image for encode_image, text prompt, and tokenizer
        image_embeds = loaded.model.encode_image(image)
        # AutoProcessor may return tokenizer directly or have .tokenizer attribute
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        return loaded.model.generate(
            image_embeds=image_embeds,
            prompt=prompt_text or "",
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
    return loaded.model.generate(**inputs, max_new_tokens=max_new_tokens)


def _run_generate_and_decode(
    loaded: LoadedModel,
    inputs: dict[str, Any],
    max_new_tokens: int = 50,
    image: "Image.Image | None" = None,
    prompt_text: str | None = None,
) -> list[str]:
    """Run generate and decode the output."""
    output = _run_generate(loaded, inputs, max_new_tokens, image=image, prompt_text=prompt_text)
    if isinstance(output, str):
        return [output]
    if isinstance(output, list) and output and isinstance(output[0], str):
        return output
    return loaded.processor.batch_decode(output, skip_special_tokens=True)


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
        is_custom = _is_custom_inference(loaded.model_name)

        # For custom inference models, only support batch_size=1
        if is_custom and batch_size > 1:
            result["status"] = "skipped"
            result["error_message"] = f"{loaded.model_name} only supports batch_size=1"
            return result

        # Prepare first sample image/prompt for custom inference
        first_img = resize_image(samples[0].image, resolution)
        first_prompt = generate_prompt(samples[0].question, prompt_length)

        if is_custom:
            inputs = {}  # Custom models don't use standard inputs
        else:
            inputs = _prepare_batch(loaded, samples, resolution, prompt_length, batch_size)

        # Latency
        def inference_fn():
            with torch.no_grad():
                _run_generate(loaded, inputs, image=first_img, prompt_text=first_prompt)

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
                    _run_generate(loaded, inputs, image=first_img, prompt_text=first_prompt)
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
            if is_custom:
                inp = {}
            else:
                inp = loaded.processor(images=img, text=prompt, return_tensors="pt")
                if loaded.device == "cuda":
                    inp = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inp.items()}
            with torch.no_grad():
                decoded = _run_generate_and_decode(loaded, inp, image=img, prompt_text=prompt)
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
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
):
    """Run the full experiment matrix from config."""
    config = load_config(config_path)
    results_path = Path(results_dir)
    completed = get_completed_experiments(results_path)

    if wandb_project:
        init_wandb(
            project=wandb_project,
            entity=wandb_entity,
            config=config,
            run_name=wandb_run_name,
        )

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
                                _log_to_wandb(result)

                # Free model memory before loading next
                del loaded
                clear_gpu_memory()

    if _wandb_run is not None:
        _wandb_run.finish()
