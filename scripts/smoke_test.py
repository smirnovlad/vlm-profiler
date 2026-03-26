#!/usr/bin/env python3
"""Smoke test: verify each model loads and generates output."""

import gc
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import format_prompt_for_model
from src.models.registry import load_model

MODELS = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/instructblip-vicuna-7b",
    "Salesforce/instructblip-flan-t5-xl",
    "Salesforce/blip2-flan-t5-xl",
    "adept/fuyu-8b",
    "HuggingFaceM4/idefics2-8b",
]

# Dummy 224x224 red image
DUMMY_IMAGE = Image.new("RGB", (224, 224), color=(255, 0, 0))
BASE_PROMPT = "What is in this image?"


def smoke_test_model(model_name: str, gpu_index: int = 0):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Load
    try:
        loaded = load_model(model_name, device="cuda", optimization="none", gpu_index=gpu_index)
        print(f"  [OK] Loaded on cuda:{gpu_index}")
    except Exception as e:
        print(f"  [FAIL] Load error: {e}")
        return False

    # Process inputs
    try:
        prompt = format_prompt_for_model(BASE_PROMPT, model_name)
        inputs = loaded.processor(images=DUMMY_IMAGE, text=prompt, return_tensors="pt")
        device_str = f"cuda:{gpu_index}"
        # Fix 5D pixel_values
        if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "dim"):
            if inputs["pixel_values"].dim() == 5:
                inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)
        inputs = {k: v.to(device_str) if hasattr(v, "to") else v for k, v in inputs.items()}
        print(f"  [OK] Processor produced keys: {list(inputs.keys())}")
    except Exception as e:
        print(f"  [FAIL] Processor error: {e}")
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
        return False

    # Generate
    try:
        with torch.no_grad():
            output_ids = loaded.model.generate(**inputs, max_new_tokens=20)
        print(f"  [OK] Generate returned shape: {output_ids.shape}")
    except Exception as e:
        print(f"  [FAIL] Generate error: {e}")
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
        return False

    # Decode
    try:
        text = loaded.processor.batch_decode(output_ids, skip_special_tokens=True)
        print(f"  [OK] Decoded: {text[0][:80]}")
    except Exception as e:
        print(f"  [FAIL] Decode error: {e}")
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
        return False

    # Cleanup
    del loaded, inputs, output_ids
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  [OK] Memory freed")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=None, help="Test specific models only")
    args = parser.parse_args()

    models = args.models or MODELS
    results = {}

    for model_name in models:
        ok = smoke_test_model(model_name, gpu_index=args.gpu_index)
        results[model_name] = "PASS" if ok else "FAIL"

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        short = name.split("/")[-1]
        print(f"  {'✓' if status == 'PASS' else '✗'} {short}: {status}")


if __name__ == "__main__":
    main()
