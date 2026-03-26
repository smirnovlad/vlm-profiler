"""FLOPs estimation for VLMs using calflops."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlopsResult:
    total_flops: float
    total_macs: float
    total_params: int
    flops_str: str
    error: str | None = None


def estimate_flops(
    model: Any,
    input_kwargs: dict[str, Any],
) -> FlopsResult:
    """Estimate FLOPs for a model forward pass.

    Uses calflops for estimation. Falls back gracefully if not supported.

    Args:
        model: The loaded model.
        input_kwargs: Dict of inputs to forward() (input_ids, pixel_values, etc.)

    Returns:
        FlopsResult with FLOPs, MACs, and parameter count.
    """
    try:
        from calflops import calculate_flops

        flops, macs, params = calculate_flops(
            model=model,
            kwargs=input_kwargs,
            print_results=False,
        )
        return FlopsResult(
            total_flops=flops,
            total_macs=macs,
            total_params=params,
            flops_str=f"{flops:.2e} FLOPs",
        )
    except Exception as e:
        logger.warning("calflops failed: %s. Falling back to parameter count.", e)
        param_count = sum(p.numel() for p in model.parameters())
        return FlopsResult(
            total_flops=0.0,
            total_macs=0.0,
            total_params=param_count,
            flops_str="N/A (calflops failed)",
            error=str(e),
        )


def count_parameters(model: Any) -> dict[str, int]:
    """Count parameters per top-level module (for bottleneck analysis)."""
    param_counts = {}
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        if count > 0:
            param_counts[name] = count
    return param_counts
