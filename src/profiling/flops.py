"""FLOPs estimation for VLMs using calflops and fvcore."""

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlopsResult:
    total_flops: float
    total_macs: float
    total_params: int
    flops_str: str
    method: str = "calflops"
    error: str | None = None


def _try_calflops(model: Any, input_kwargs: dict[str, Any]) -> FlopsResult | None:
    """Try calflops with output_as_string=False to get numeric values."""
    try:
        from calflops import calculate_flops

        flops, macs, params = calculate_flops(
            model=model,
            kwargs=input_kwargs,
            print_results=False,
            output_as_string=False,
        )
        if isinstance(flops, (int, float)) and flops > 0:
            return FlopsResult(
                total_flops=float(flops),
                total_macs=float(macs),
                total_params=int(params),
                flops_str=f"{flops:.2e} FLOPs",
                method="calflops",
            )
    except Exception as e:
        logger.debug("calflops failed: %s", e)
    return None


def _try_torch_profiler(model: Any, input_kwargs: dict[str, Any]) -> FlopsResult | None:
    """Estimate FLOPs using torch.utils.flop_counter."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                model(**input_kwargs)
        total_flops = flop_counter.get_total_flops()
        if total_flops > 0:
            param_count = sum(p.numel() for p in model.parameters())
            return FlopsResult(
                total_flops=float(total_flops),
                total_macs=float(total_flops) / 2.0,
                total_params=param_count,
                flops_str=f"{total_flops:.2e} FLOPs",
                method="torch_flop_counter",
            )
    except Exception as e:
        logger.debug("torch flop_counter failed: %s", e)
    return None


def _estimate_from_params(model: Any) -> FlopsResult:
    """Fallback: estimate FLOPs as ~2 * params (rough approximation for one forward pass)."""
    param_count = sum(p.numel() for p in model.parameters())
    estimated_flops = param_count * 2.0
    return FlopsResult(
        total_flops=estimated_flops,
        total_macs=float(param_count),
        total_params=param_count,
        flops_str=f"~{estimated_flops:.2e} FLOPs (estimated from params)",
        method="param_estimate",
        error="Exact FLOPs unavailable, estimated as 2 * num_params",
    )


def estimate_flops(model: Any, input_kwargs: dict[str, Any]) -> FlopsResult:
    """Estimate FLOPs for a model forward pass.

    Tries calflops first, then torch flop_counter, then falls back to param estimate.
    """
    # Try calflops
    result = _try_calflops(model, input_kwargs)
    if result is not None:
        logger.info("FLOPs via calflops: %s", result.flops_str)
        return result

    # Try torch profiler
    result = _try_torch_profiler(model, input_kwargs)
    if result is not None:
        logger.info("FLOPs via torch flop_counter: %s", result.flops_str)
        return result

    # Fallback to param estimate
    result = _estimate_from_params(model)
    logger.warning("FLOPs estimated from params: %s", result.flops_str)
    return result


def count_parameters(model: Any) -> dict[str, int]:
    """Count parameters per top-level module (for bottleneck analysis)."""
    param_counts = {}
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        if count > 0:
            param_counts[name] = count
    return param_counts
