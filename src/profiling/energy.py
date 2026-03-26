"""GPU energy consumption measurement via nvidia-smi polling."""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnergyResult:
    avg_power_w: float
    max_power_w: float
    min_power_w: float
    duration_s: float
    energy_j: float
    power_readings: list[float] = field(default_factory=list)


class EnergyMonitor:
    """Background thread that polls nvidia-smi for GPU power draw."""

    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.1):
        self._gpu_index = gpu_index
        self._poll_interval = poll_interval_s
        self._readings: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    def _poll_power(self) -> float | None:
        """Read current power draw in watts from nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._gpu_index}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, OSError) as e:
            logger.debug("Power poll failed: %s", e)
        return None

    def _monitor_loop(self):
        while self._running:
            power = self._poll_power()
            if power is not None:
                self._readings.append(power)
            time.sleep(self._poll_interval)

    def start(self):
        """Start background power monitoring."""
        self._readings = []
        self._running = True
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> EnergyResult:
        """Stop monitoring and return energy results."""
        self._running = False
        self._end_time = time.perf_counter()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        duration = self._end_time - self._start_time

        if not self._readings:
            logger.warning("No power readings collected.")
            return EnergyResult(
                avg_power_w=0.0,
                max_power_w=0.0,
                min_power_w=0.0,
                duration_s=duration,
                energy_j=0.0,
            )

        avg_power = sum(self._readings) / len(self._readings)
        return EnergyResult(
            avg_power_w=avg_power,
            max_power_w=max(self._readings),
            min_power_w=min(self._readings),
            duration_s=duration,
            energy_j=avg_power * duration,
            power_readings=list(self._readings),
        )
