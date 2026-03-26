"""Utilities: GPU management, logging, OOM handling."""

import gc
import logging
import subprocess
import sys

import torch


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_gpu_memory_mb(device_index: int = 0) -> dict[str, float]:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated(device_index) / 1024**2,
        "reserved": torch.cuda.memory_reserved(device_index) / 1024**2,
        "total": torch.cuda.get_device_properties(device_index).total_mem / 1024**2,
    }


def clear_gpu_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_my_gpu_processes() -> list[dict]:
    """List GPU processes owned by the current user."""
    import os

    my_uid = os.getuid()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        processes = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 1:
                pid = int(parts[0])
                # Check if the process belongs to us
                try:
                    stat_path = f"/proc/{pid}/status"
                    with open(stat_path) as f:
                        for status_line in f:
                            if status_line.startswith("Uid:"):
                                proc_uid = int(status_line.split()[1])
                                if proc_uid == my_uid:
                                    processes.append({"pid": pid, "info": line})
                                break
                except (FileNotFoundError, PermissionError):
                    pass
        return processes
    except Exception:
        return []


def kill_my_gpu_processes():
    """Kill GPU processes owned by the current user."""
    import os
    import signal

    procs = get_my_gpu_processes()
    for proc in procs:
        pid = proc["pid"]
        if pid != os.getpid():
            try:
                os.kill(pid, signal.SIGTERM)
                logging.getLogger(__name__).info("Killed GPU process %d", pid)
            except ProcessLookupError:
                pass


def is_valid_combo(device: str, optimization: str) -> bool:
    """Check if a device+optimization combination is valid."""
    invalid = {
        ("cpu", "fp16"),
        ("cpu", "flash_attn2"),
        ("cpu", "torch_compile"),
    }
    return (device, optimization) not in invalid
