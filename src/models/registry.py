"""Model registry: unified interface to load and configure VLMs."""

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor

# Patch Module.__getattr__ to handle missing all_tied_weights_keys
# (needed for models with custom remote code on transformers 5.x)
_orig_module_getattr = nn.Module.__getattr__


def _patched_module_getattr(self, name: str):
    if name == "all_tied_weights_keys":
        return getattr(self, "_tied_weights_keys", {}) or {}
    return _orig_module_getattr(self, name)


nn.Module.__getattr__ = _patched_module_getattr

logger = logging.getLogger(__name__)

# Models that require trust_remote_code=True
TRUST_REMOTE_CODE_MODELS = {
    "THUDM/cogagent-vqa-hf",
    "adept/fuyu-8b",
    "vikhyatk/moondream2",
}

# Model-specific class overrides (when AutoModel doesn't work)
MODEL_CLASS_OVERRIDES: dict[str, tuple[str, str]] = {
    # (model_class_module, model_class_name)
    "Salesforce/blip2-opt-2.7b": (
        "transformers",
        "Blip2ForConditionalGeneration",
    ),
    "Salesforce/blip2-flan-t5-xl": (
        "transformers",
        "Blip2ForConditionalGeneration",
    ),
    "Salesforce/instructblip-vicuna-7b": (
        "transformers",
        "InstructBlipForConditionalGeneration",
    ),
    "Salesforce/instructblip-flan-t5-xl": (
        "transformers",
        "InstructBlipForConditionalGeneration",
    ),
    "llava-hf/llava-1.5-7b-hf": (
        "transformers",
        "LlavaForConditionalGeneration",
    ),
    "llava-hf/llava-1.5-13b-hf": (
        "transformers",
        "LlavaForConditionalGeneration",
    ),
    "HuggingFaceM4/idefics2-8b": (
        "transformers",
        "Idefics2ForConditionalGeneration",
    ),
    "adept/fuyu-8b": (
        "transformers",
        "FuyuForCausalLM",
    ),
}

# Models that use 'dtype' kwarg instead of 'torch_dtype'
DTYPE_KWARG_MODELS = {
    "vikhyatk/moondream2",
}

# Models that don't support device_map and need explicit .to(device)
NO_DEVICE_MAP_MODELS = {
    "vikhyatk/moondream2",
}

# Models that should always load in FP16 (too large for FP32 on 48GB GPU)
FP16_DEFAULT_MODELS = {
    "llava-hf/llava-1.5-13b-hf",
}


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    model_name: str
    device: str
    dtype: torch.dtype
    optimization: str
    gpu_index: int = 0


def _get_model_class(model_name: str):
    """Get the correct model class for a given model."""
    if model_name in MODEL_CLASS_OVERRIDES:
        module_name, class_name = MODEL_CLASS_OVERRIDES[model_name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    return AutoModelForCausalLM


def load_model(
    model_name: str,
    device: str = "cuda",
    optimization: str = "none",
    gpu_index: int = 0,
) -> LoadedModel:
    """Load a VLM with the specified optimization.

    Args:
        model_name: HuggingFace model identifier
        device: 'cuda' or 'cpu'
        optimization: 'none', 'fp16', 'torch_compile', 'flash_attn2'
        gpu_index: GPU device index (0 or 1) for multi-GPU setups

    Returns:
        LoadedModel with model, processor, and metadata
    """
    trust_remote = model_name in TRUST_REMOTE_CODE_MODELS
    dtype = torch.float32
    uses_dtype_kwarg = model_name in DTYPE_KWARG_MODELS
    no_device_map = model_name in NO_DEVICE_MAP_MODELS

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote,
    }

    if not no_device_map:
        device_target = f"cuda:{gpu_index}" if device == "cuda" else None
        model_kwargs["device_map"] = device_target

    dtype_key = "dtype" if uses_dtype_kwarg else "torch_dtype"

    force_fp16 = model_name in FP16_DEFAULT_MODELS

    if optimization == "fp16" or force_fp16:
        dtype = torch.float16
        model_kwargs[dtype_key] = torch.float16
    elif optimization == "flash_attn2":
        dtype = torch.float16
        model_kwargs[dtype_key] = torch.float16
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs[dtype_key] = torch.float32

    logger.info(
        "Loading model=%s, device=%s, opt=%s", model_name, device, optimization
    )

    model_class = _get_model_class(model_name)
    model = model_class.from_pretrained(model_name, **model_kwargs)

    # Models without device_map need explicit .to()
    if no_device_map and device == "cuda":
        model = model.to(f"cuda:{gpu_index}")
    elif device == "cpu":
        model = model.to("cpu")

    if optimization == "torch_compile":
        logger.info("Applying torch.compile to %s", model_name)
        model = torch.compile(model)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote
    )

    return LoadedModel(
        model=model,
        processor=processor,
        model_name=model_name,
        device=device,
        dtype=dtype,
        optimization=optimization,
        gpu_index=gpu_index,
    )
