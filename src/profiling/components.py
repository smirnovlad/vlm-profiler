"""Per-component latency and prefill/decode breakdown.

Two orthogonal measurements:

1) ComponentTimer — CUDA-event forward hooks on top-level submodules
   (vision_model, qformer, language_model, etc.). Aggregates time spent
   inside each submodule across every call during one generate().

2) measure_prefill_decode — estimates prefill and per-token decode cost by
   running generate() with two different max_new_tokens values and solving
   for the two unknowns. Assumes decode time per token is roughly constant.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


# Top-level submodule names we probe, in display order.
# Not all models have all of these; missing ones are skipped silently.
COMPONENT_NAMES: tuple[str, ...] = (
    "vision_model",            # BLIP-2, InstructBLIP
    "vision_tower",            # LLaVA
    "vision_embed_tokens",     # Fuyu
    "qformer",                 # BLIP-2, InstructBLIP
    "multi_modal_projector",   # LLaVA
    "language_model",          # BLIP-2, InstructBLIP, LLaVA
    "language_projection",     # BLIP-2 (small proj layer)
)


@dataclass(frozen=True)
class ComponentTimes:
    per_call_ms: dict[str, float]       # average ms per component per generate() call
    per_call_share: dict[str, float]    # fraction of total measured component time
    total_measured_ms: float            # sum over all components (may be < wall time)
    wall_ms: float                      # wall-clock time of one generate()


class ComponentTimer:
    """Context manager that records per-submodule CUDA event times."""

    def __init__(self, model: Any, component_names: tuple[str, ...] = COMPONENT_NAMES):
        self._model = model
        self._names = component_names
        self._starts: dict[str, list[torch.cuda.Event]] = {}
        self._ends: dict[str, list[torch.cuda.Event]] = {}
        self._handles: list = []
        # Resolve submodules (may be attrs on the wrapper or on an inner `.model`)
        self._resolved: dict[str, torch.nn.Module] = {}
        for name in component_names:
            mod = _resolve_submodule(model, name)
            if mod is not None:
                self._resolved[name] = mod

    def __enter__(self) -> "ComponentTimer":
        for name, mod in self._resolved.items():
            self._starts[name] = []
            self._ends[name] = []

            def make_pre(n: str):
                def pre(_mod, _inp):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    self._starts[n].append(ev)
                return pre

            def make_post(n: str):
                def post(_mod, _inp, _out):
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    self._ends[n].append(ev)
                return post

            self._handles.append(mod.register_forward_pre_hook(make_pre(name)))
            self._handles.append(mod.register_forward_hook(make_post(name)))
        return self

    def __exit__(self, *_exc) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def collect_ms(self) -> dict[str, float]:
        """Synchronize and return accumulated ms per component for this context."""
        torch.cuda.synchronize()
        out: dict[str, float] = {}
        for name, starts in self._starts.items():
            ends = self._ends.get(name, [])
            if not starts or len(starts) != len(ends):
                continue
            total = 0.0
            for s, e in zip(starts, ends):
                total += s.elapsed_time(e)
            out[name] = total
        return out


def _resolve_submodule(model: Any, name: str) -> torch.nn.Module | None:
    """Look up a named submodule on the model, or on a common `.model` wrapper."""
    direct = getattr(model, name, None)
    if isinstance(direct, torch.nn.Module):
        return direct
    inner = getattr(model, "model", None)
    if inner is not None:
        nested = getattr(inner, name, None)
        if isinstance(nested, torch.nn.Module):
            return nested
    return None


def measure_component_latency(
    model: Any,
    inputs: dict[str, Any],
    generate_fn,
    warmup_runs: int = 2,
    timed_runs: int = 3,
) -> ComponentTimes:
    """Average component latency over `timed_runs` generate calls.

    Args:
        model: The underlying VLM.
        inputs: Already-prepared input dict for model.generate().
        generate_fn: Callable(inputs) -> Tensor that wraps model.generate().
                      Taking it as a callable lets the caller use any
                      model-specific generate signature.
        warmup_runs: CUDA warmup passes (not counted).
        timed_runs: Counted passes; results are averaged.
    """
    for _ in range(warmup_runs):
        with torch.no_grad():
            generate_fn(inputs)
    torch.cuda.synchronize()

    per_call_sum: dict[str, float] = {}
    wall_sum = 0.0
    for _ in range(timed_runs):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        with ComponentTimer(model) as timer:
            with torch.no_grad():
                generate_fn(inputs)
        ms = timer.collect_ms()
        wall_sum += (time.perf_counter() - wall_start) * 1000.0
        for k, v in ms.items():
            per_call_sum[k] = per_call_sum.get(k, 0.0) + v

    if timed_runs == 0:
        return ComponentTimes({}, {}, 0.0, 0.0)

    per_call = {k: v / timed_runs for k, v in per_call_sum.items()}
    total_measured = sum(per_call.values())
    share = (
        {k: v / total_measured for k, v in per_call.items()}
        if total_measured > 0
        else {k: 0.0 for k in per_call}
    )
    return ComponentTimes(
        per_call_ms=per_call,
        per_call_share=share,
        total_measured_ms=total_measured,
        wall_ms=wall_sum / timed_runs,
    )


@dataclass(frozen=True)
class PrefillDecodeTimes:
    prefill_ms: float                # prefill + 1 decode step (approx prefill)
    decode_per_token_ms: float       # marginal per-token autoregressive cost
    total_at_50_ms: float             # measured wall time for max_new_tokens=50
    short_n: int
    long_n: int


def measure_prefill_decode(
    generate_fn,
    inputs: dict[str, Any],
    short_n: int = 1,
    long_n: int = 50,
    warmup_runs: int = 1,
    timed_runs: int = 3,
) -> PrefillDecodeTimes:
    """Separate prefill cost from per-token decode cost.

    Runs generate() with max_new_tokens=short_n and max_new_tokens=long_n,
    then solves: total(N) ≈ prefill + N * decode_per_token.

    short_n=1 gives prefill + 1 decode step (≈ prefill for small models,
    but we report it as-is and let the consumer interpret).
    """

    def _time_runs(n: int) -> float:
        for _ in range(warmup_runs):
            with torch.no_grad():
                generate_fn(inputs, max_new_tokens=n)
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(timed_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                generate_fn(inputs, max_new_tokens=n)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return sum(times) / len(times)

    short_ms = _time_runs(short_n)
    long_ms = _time_runs(long_n)
    span = max(long_n - short_n, 1)
    decode_per_token = max((long_ms - short_ms) / span, 0.0)
    prefill_ms = max(short_ms - short_n * decode_per_token, 0.0)
    return PrefillDecodeTimes(
        prefill_ms=prefill_ms,
        decode_per_token_ms=decode_per_token,
        total_at_50_ms=long_ms,
        short_n=short_n,
        long_n=long_n,
    )
