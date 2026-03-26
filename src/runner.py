"""Main experiment runner: vary-one-axis design for efficient profiling."""

import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.loader import VLMSample, load_dataset_by_name
from src.data.preprocessing import format_prompt_for_model, generate_prompt, resize_image
from src.models.registry import LoadedModel, load_model
from src.profiling.energy import EnergyMonitor
from src.profiling.flops import count_parameters, estimate_flops
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
            log_data["energy/per_inference_j"] = result["energy"].get("energy_per_inference_j", 0)
            log_data["energy/total_j"] = result["energy"]["energy_j"]
        if result.get("quality"):
            log_data["quality/wer"] = result["quality"]["wer_score"]
            log_data["quality/exact_match"] = result["quality"]["exact_match_accuracy"]
            log_data["quality/num_samples"] = result["quality"]["num_samples"]
        if result.get("flops"):
            log_data["flops/total"] = result["flops"]["total_flops"]
            log_data["flops/macs"] = result["flops"]["total_macs"]
            log_data["flops/params"] = result["flops"]["total_params"]
            log_data["flops/method"] = result["flops"].get("method", "unknown")
        if result.get("param_counts"):
            for component, count in result["param_counts"].items():
                log_data[f"params/{component}"] = count

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
    prompts = [
        format_prompt_for_model(generate_prompt(s.question, prompt_length), loaded.model_name)
        for s in batch_samples
    ]

    if batch_size == 1:
        inputs = loaded.processor(
            images=images[0], text=prompts[0], return_tensors="pt"
        )
    else:
        _set_pad_token(loaded.processor)
        inputs = loaded.processor(
            images=images, text=prompts, return_tensors="pt", padding=True
        )

    # Fix 5D pixel_values (some processors add an extra dim)
    if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "dim"):
        if inputs["pixel_values"].dim() == 5 and inputs["pixel_values"].shape[1] == 1:
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)

    device_str = f"cuda:{loaded.gpu_index}" if loaded.device == "cuda" else "cpu"
    inputs = {k: v.to(device_str) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


def _run_generate(
    loaded: LoadedModel,
    inputs: dict[str, Any],
    max_new_tokens: int = 50,
    image=None,
    prompt_text: str | None = None,
):
    """Run model.generate() with model-specific handling."""
    if _is_custom_inference(loaded.model_name) and image is not None:
        image_embeds = loaded.model.encode_image(image)
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
    image=None,
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
    measure_quality: bool = True,
    measure_flops: bool = False,
    quality_samples: int | None = None,
) -> dict[str, Any]:
    """Run a single experiment configuration and return results dict.

    Args:
        measure_quality: If False, skip quality evaluation (saves time for scaling experiments).
        measure_flops: If True, estimate FLOPs for this config.
        quality_samples: Max samples for quality eval. None = use all.
    """
    gpu_index = loaded.gpu_index
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

        if is_custom and batch_size > 1:
            result["status"] = "skipped"
            result["error_message"] = f"{loaded.model_name} only supports batch_size=1"
            return result

        first_img = resize_image(samples[0].image, resolution)
        first_prompt = format_prompt_for_model(
            generate_prompt(samples[0].question, prompt_length), loaded.model_name
        )

        if is_custom:
            inputs = {}
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

        # Energy (per-inference: total energy / number of runs)
        if loaded.device == "cuda":
            monitor = EnergyMonitor(gpu_index=gpu_index)
            monitor.start()
            for _ in range(timed_runs):
                with torch.no_grad():
                    _run_generate(loaded, inputs, image=first_img, prompt_text=first_prompt)
                torch.cuda.synchronize()
            energy_raw = monitor.stop()
            energy_data = asdict(energy_raw)
            energy_data["energy_per_inference_j"] = energy_raw.energy_j / timed_runs
            energy_data["duration_per_inference_s"] = energy_raw.duration_s / timed_runs
            energy_data["num_runs"] = timed_runs
            result["energy"] = energy_data
        else:
            result["energy"] = None

        # Parameter counts (always — cheap and needed for bottleneck analysis)
        if not is_custom:
            result["param_counts"] = count_parameters(loaded.model)
        else:
            result["param_counts"] = None

        # FLOPs (optional, run once on baseline)
        if measure_flops and not is_custom and inputs:
            try:
                flops_result = estimate_flops(loaded.model, inputs)
                result["flops"] = asdict(flops_result)
            except Exception as e:
                logger.warning("FLOPs estimation failed for %s: %s", exp_id, e)
                result["flops"] = None
        else:
            result["flops"] = None

        # Quality (optional — skip for scaling experiments where quality doesn't change)
        if measure_quality:
            eval_samples = samples
            if quality_samples is not None:
                eval_samples = samples[:quality_samples]

            predictions = []
            references = []
            for sample in eval_samples:
                img = resize_image(sample.image, resolution)
                prompt = format_prompt_for_model(
                    generate_prompt(sample.question, prompt_length), loaded.model_name
                )
                if is_custom:
                    inp = {}
                else:
                    inp = loaded.processor(images=img, text=prompt, return_tensors="pt")
                    if "pixel_values" in inp and hasattr(inp["pixel_values"], "dim"):
                        if inp["pixel_values"].dim() == 5 and inp["pixel_values"].shape[1] == 1:
                            inp["pixel_values"] = inp["pixel_values"].squeeze(1)
                    device_str = f"cuda:{gpu_index}" if loaded.device == "cuda" else "cpu"
                    inp = {k: v.to(device_str) if hasattr(v, "to") else v for k, v in inp.items()}
                with torch.no_grad():
                    decoded = _run_generate_and_decode(loaded, inp, image=img, prompt_text=prompt)
                predictions.append(decoded[0] if decoded else "")
                references.append(sample.answer)

            quality = compute_quality(predictions, references)
            result["quality"] = asdict(quality)
        else:
            result["quality"] = None

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
                if data.get("status") in ("success", "oom", "skipped"):
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


def _generate_experiment_configs(config: dict) -> list[dict]:
    """Generate vary-one-axis experiment configs instead of full cartesian product.

    Design:
      1. Baseline: default resolution, prompt_length, batch_size, optimization per dataset
         → measures quality + latency + energy + FLOPs
      2. Resolution sweep: vary resolution, fix everything else at baseline
         → latency + energy only (quality doesn't change meaningfully)
      3. Prompt length sweep: vary prompt_length, fix everything else
         → latency + energy only
      4. Batch size sweep: vary batch_size, fix everything else
         → latency + energy only
      5. Optimization sweep: vary optimization, fix everything else
         → latency + energy only
    """
    scaling = config["scaling"]
    baseline = config.get("baseline", {})
    base_res = baseline.get("resolution", scaling["resolutions"][0])
    base_plen = baseline.get("prompt_length", scaling["prompt_lengths"][0])
    base_bs = baseline.get("batch_size", scaling["batch_sizes"][0])
    base_opt = baseline.get("optimization", "none")
    base_device = baseline.get("device", "cuda")

    experiments = []
    seen = set()

    def add(res, plen, bs, opt, device, quality, flops):
        key = (res, plen, bs, opt, device)
        if key not in seen and is_valid_combo(device, opt):
            seen.add(key)
            experiments.append({
                "resolution": res,
                "prompt_length": plen,
                "batch_size": bs,
                "optimization": opt,
                "device": device,
                "measure_quality": quality,
                "measure_flops": flops,
            })

    # 1. Baseline (quality + flops)
    add(base_res, base_plen, base_bs, base_opt, base_device, quality=True, flops=True)

    # 2. Resolution sweep
    for res in scaling["resolutions"]:
        add(res, base_plen, base_bs, base_opt, base_device, quality=False, flops=False)

    # 3. Prompt length sweep
    for plen in scaling["prompt_lengths"]:
        add(base_res, plen, base_bs, base_opt, base_device, quality=False, flops=False)

    # 4. Batch size sweep
    for bs in scaling["batch_sizes"]:
        add(base_res, base_plen, bs, base_opt, base_device, quality=False, flops=False)

    # 5. Optimization sweep
    for opt in scaling["optimizations"]:
        add(base_res, base_plen, base_bs, opt, base_device, quality=False, flops=False)

    # 6. CPU baseline (if configured)
    if "cpu" in scaling.get("devices", []):
        add(base_res, base_plen, base_bs, "none", "cpu", quality=False, flops=False)

    return experiments


def run_experiments(
    config_path: str,
    results_dir: str = "results",
    models_filter: list[str] | None = None,
    datasets_filter: list[str] | None = None,
    devices_filter: list[str] | None = None,
    gpu_index: int = 0,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
):
    """Run vary-one-axis experiments from config."""
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

    # Generate experiment configs (vary-one-axis)
    exp_configs = _generate_experiment_configs(config)
    logger.info(
        "Generated %d experiment configs per model per dataset (vary-one-axis design)",
        len(exp_configs),
    )

    for model_name in config["models"]:
        if models_filter and model_name not in models_filter:
            continue

        # Group experiments by (optimization, device) to minimize model reloads
        combos: dict[tuple[str, str], list[dict]] = {}
        for ec in exp_configs:
            key = (ec["optimization"], ec["device"])
            if devices_filter and ec["device"] not in devices_filter:
                continue
            combos.setdefault(key, []).append(ec)

        for (optimization, device), configs_for_combo in combos.items():
            try:
                loaded = load_model(model_name, device, optimization, gpu_index=gpu_index)
            except Exception as e:
                logger.error(
                    "Failed to load %s (opt=%s, dev=%s): %s",
                    model_name, optimization, device, e,
                )
                continue

            for ds_name, samples in dataset_cache.items():
                for ec in configs_for_combo:
                    exp_id = _make_experiment_id(
                        model_name, ds_name, ec["resolution"],
                        ec["prompt_length"], device, optimization, ec["batch_size"],
                    )
                    if exp_id in completed:
                        logger.info("Skipping (done): %s", exp_id)
                        continue

                    result = run_single_experiment(
                        loaded, samples,
                        ec["resolution"], ec["prompt_length"], ec["batch_size"],
                        warmup_runs=config["profiling"]["warmup_runs"],
                        timed_runs=config["profiling"]["timed_runs"],
                        measure_quality=ec["measure_quality"],
                        measure_flops=ec["measure_flops"],
                    )
                    save_result(result, results_path)
                    _log_to_wandb(result)

            del loaded
            clear_gpu_memory()

    if _wandb_run is not None:
        _wandb_run.finish()
