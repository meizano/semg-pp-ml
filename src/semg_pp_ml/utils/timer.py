"""
Timer utilities for performance tracking.
"""
from __future__ import annotations

import time
import logging
from typing import Dict
from contextlib import contextmanager


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, stage_name: str, log_dict: Dict[str, float]):
        self.name = stage_name
        self.log = log_dict

    def __enter__(self):
        self.t0 = time.perf_counter()
        logging.info(f"\n{'='*70}\n[STAGE] {self.name}\n{'='*70}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.t0
        self.log[self.name] = elapsed
        logging.info(f"[DONE] {self.name}: {elapsed:.4f}s ({elapsed/60:.2f} min)")
        return False


@contextmanager
def timer(stage_name: str, log_dict: Dict[str, float]):
    """Function-based timer context manager."""
    t = Timer(stage_name, log_dict)
    t.__enter__()
    try:
        yield t
    finally:
        t.__exit__(None, None, None)


class EpochTimer:
    """Callback-compatible epoch timer for Keras."""

    def __init__(self):
        self.epoch_times = []
        self.throughputs = []
        self._epoch_start = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self._epoch_start
        self.epoch_times.append(elapsed)
        samples = logs.get('samples', 0) if logs else 0
        self.throughputs.append(samples / (elapsed + 1e-10))

    @property
    def avg_time(self) -> float:
        return np.mean(self.epoch_times) if self.epoch_times else 0

    @property
    def avg_throughput(self) -> float:
        return np.mean(self.throughputs) if self.throughputs else 0