"""
Real-time performance and resource benchmarking utility (WPS, Latency, CPU, RAM).
"""
from __future__ import annotations

import time
import psutil
import numpy as np
from typing import Dict, Any, Optional


def benchmark_model_inference(
    model: Any,
    X_test: np.ndarray,
    batch_size: int = 32,
    n_warmup: int = 2
) -> Dict[str, float]:
    """
    Benchmark inference speed (WPS / FPS, latency) and resource usage (CPU %, RAM MB).
    """
    process = psutil.Process()
    
    # Baseline memory and CPU
    mem_before = process.memory_info().rss / (1024 * 1024)
    cpu_before = process.cpu_percent(interval=0.1)

    # Warmup
    if len(X_test) > 0:
        for _ in range(n_warmup):
            _ = model.predict_proba(X_test[:min(batch_size, len(X_test))])

    t0 = time.perf_counter()
    if hasattr(model, 'predict_proba'):
        _ = model.predict_proba(X_test)
    else:
        _ = model.predict(X_test)
    t_elapsed = time.perf_counter() - t0

    total_samples = len(X_test)
    wps = total_samples / (t_elapsed + 1e-10)
    latency_ms = (t_elapsed / max(1, total_samples)) * 1000.0

    mem_after = process.memory_info().rss / (1024 * 1024)
    cpu_after = process.cpu_percent(interval=0.1)
    cpu_usage = max(cpu_after, psutil.cpu_percent(interval=None))

    return {
        "total_samples": total_samples,
        "inference_time_sec": round(t_elapsed, 4),
        "windows_per_second_wps": round(wps, 1),
        "latency_ms_per_window": round(latency_ms, 4),
        "cpu_usage_percent": round(cpu_usage, 1),
        "ram_footprint_mb": round(mem_after, 2),
        "ram_delta_mb": round(mem_after - mem_before, 2)
    }
