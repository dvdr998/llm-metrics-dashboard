import platform
import psutil
import torch


def get_best_device() -> str:
    """
    Chooses the best available device.

    cuda = NVIDIA GPU
    mps = Apple Silicon GPU
    cpu = fallback
    """
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_device_snapshot() -> dict:
    """
    Captures current CPU, RAM, and GPU-related information.
    This works on Mac, CPU, and NVIDIA GPU systems.
    """
    device = get_best_device()

    snapshot = {
        "device": device,
        "platform": platform.platform(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 3),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 3),
    }

    if device == "cuda":
        snapshot.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 3),
            "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 3),
            "cuda_peak_memory_gb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3),
        })

    elif device == "mps":
        snapshot.update({
            "gpu_name": "Apple Silicon GPU using MPS",
            "mps_available": True,
        })

        try:
            snapshot["mps_memory_allocated_gb"] = round(
                torch.mps.current_allocated_memory() / (1024 ** 3), 3
            )
        except Exception:
            snapshot["mps_memory_allocated_gb"] = "Not available"

    else:
        snapshot.update({
            "gpu_name": "No GPU acceleration detected",
        })

    return snapshot


def reset_peak_memory():
    """
    Resets peak memory tracking when supported.
    """
    device = get_best_device()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
