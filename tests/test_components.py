"""Smoke test for ComponentTimer and measure_prefill_decode on a tiny model.

Runs on CUDA only. Uses a trivial nn.Module with the expected submodule
names so we exercise the hook plumbing without loading a real VLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.profiling.components import (  # noqa: E402
    ComponentTimer,
    measure_component_latency,
    measure_prefill_decode,
)


class _TinyVLM(nn.Module):
    """Fake VLM: has vision_model and language_model top-level children."""

    def __init__(self):
        super().__init__()
        self.vision_model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        self.language_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.language_model(self.vision_model(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_component_timer_captures_both_submodules():
    device = torch.device("cuda:0")
    model = _TinyVLM().to(device).eval()
    x = torch.randn(1, 16, device=device)

    with ComponentTimer(model) as timer:
        with torch.no_grad():
            for _ in range(5):
                model(x)

    ms = timer.collect_ms()
    assert "vision_model" in ms
    assert "language_model" in ms
    assert ms["vision_model"] > 0.0
    assert ms["language_model"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_measure_component_latency_shares_sum_to_one():
    device = torch.device("cuda:0")
    model = _TinyVLM().to(device).eval()
    x = torch.randn(1, 16, device=device)

    def gen(_inputs):
        return model(x)

    result = measure_component_latency(model, {}, gen, warmup_runs=1, timed_runs=2)
    assert abs(sum(result.per_call_share.values()) - 1.0) < 1e-6
    assert result.wall_ms > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_decode_non_negative():
    device = torch.device("cuda:0")
    model = _TinyVLM().to(device).eval()
    x = torch.randn(1, 16, device=device)

    def gen(_inputs, max_new_tokens: int = 1):
        # Simulate autoregressive cost: run more passes for larger N
        out = None
        for _ in range(max_new_tokens):
            out = model(x)
        return out

    result = measure_prefill_decode(gen, {}, short_n=1, long_n=8, warmup_runs=1, timed_runs=2)
    assert result.prefill_ms >= 0.0
    assert result.decode_per_token_ms >= 0.0
