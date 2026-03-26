"""Model registry: unified interface to load and configure VLMs."""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

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
    "llava-hf/llava-1.5-400m-fp16": (
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


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    model_name: str
    device: str
    dtype: torch.dtype
    optimization: str


def _get_model_class(model_name: str):
    """Get the correct model class for a given model."""
    if model_name in MODEL_CLASS_OVERRIDES:
        module_name, class_name = MODEL_CLASS_OVERRIDES[model_name]
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    return AutoModelForCausalLM


def load_model(
    model_name: str,
    device: str = "cuda",
    optimization: str = "none",
) -> LoadedModel:
    """Load a VLM with the specified optimization.

    Args:
        model_name: HuggingFace model identifier
        device: 'cuda' or 'cpu'
        optimization: 'none', 'fp16', 'torch_compile', 'flash_attn2'

    Returns:
        LoadedModel with model, processor, and metadata
    """
    trust_remote = model_name in TRUST_REMOTE_CODE_MODELS
    dtype = torch.float32

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote,
        "device_map": device if device == "cuda" else None,
    }

    if optimization == "fp16":
        dtype = torch.float16
        model_kwargs["torch_dtype"] = torch.float16
    elif optimization == "flash_attn2":
        dtype = torch.float16
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    logger.info(
        "Loading model=%s, device=%s, opt=%s", model_name, device, optimization
    )

    model_class = _get_model_class(model_name)
    model = model_class.from_pretrained(model_name, **model_kwargs)

    if device == "cpu":
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
    )
