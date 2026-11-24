"""Label-preserving augmentations for sEMG windows.

All functions take windows shaped ``(n_windows, n_samples, n_channels)`` and
return an array of the same shape (except ``mixup_same_class``, which keeps the
shape but needs labels). Randomness is controlled by a ``numpy.random.Generator``
so results are reproducible.

Ideas are taken from common sEMG augmentation practice reported in the
repositories surveyed in ``docs/related_work.md``:
- time warping / slicing and interpolation (Yazhou-Z/EMG_CNN)
- additive noise, amplitude scaling, electrode shift (sEMG-CNNTransformer, CIMS dual-stream CNN)
- baseline wander and powerline interference (real-world recording artifacts)
"""

from __future__ import annotations

import numpy as np


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _check_windows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected windows of shape (n_windows, n_samples, n_channels), got {x.shape}")
    return x


def add_gaussian_noise(x, snr_db: float = 20.0, rng=None) -> np.ndarray:
    """Add white noise so that each channel of each window has the requested SNR (dB)."""
    x = _check_windows(x)
    rng = _rng(rng)
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True)) + 1e-12
    noise_std = rms / (10 ** (snr_db / 20.0))
    return x + rng.standard_normal(x.shape).astype(np.float32) * noise_std.astype(np.float32)


def random_amplitude_scale(x, sigma: float = 0.15, rng=None) -> np.ndarray:
    """Multiply each (window, channel) by a log-normal gain, mimicking electrode impedance/skin changes."""
    x = _check_windows(x)
    rng = _rng(rng)
    gains = np.exp(rng.normal(0.0, sigma, size=(x.shape[0], 1, x.shape[2]))).astype(np.float32)
    return x * gains


def time_warp(x, sigma: float = 0.2, n_knots: int = 4, rng=None) -> np.ndarray:
    """Smooth random local time-stretching. Output length equals input length.

    A random speed profile (drawn at ``n_knots`` points and linearly interpolated)
    re-samples each window, so contraction speed varies within the window.
    """
    x = _check_windows(x)
    rng = _rng(rng)
    n, T, C = x.shape
    out = np.empty_like(x)
    t = np.arange(T, dtype=np.float64)
    knots = np.linspace(0, T - 1, n_knots + 2)
    for i in range(n):
        speeds = np.clip(rng.normal(1.0, sigma, size=n_knots + 2), 0.2, None)
        speed = np.interp(t, knots, speeds)
        src = np.cumsum(speed)
        src = (src - src[0]) / (src[-1] - src[0]) * (T - 1)
        for c in range(C):
            out[i, :, c] = np.interp(src, t, x[i, :, c])
    return out


def electrode_shift(x, max_shift: int = 1, rng=None) -> np.ndarray:
    """Shift the electrode array by a random integer number of channels per window.

    Models electrode placement drift around the muscle. Edge channels are replicated.
    """
    x = _check_windows(x)
    rng = _rng(rng)
    n, _, C = x.shape
    out = np.empty_like(x)
    shifts = rng.integers(-max_shift, max_shift + 1, size=n)
    base = np.arange(C)
    for i, s in enumerate(shifts):
        idx = np.clip(base + s, 0, C - 1)
        out[i] = x[i][:, idx]
    return out


def crosstalk(x, max_alpha: float = 0.2, rng=None) -> np.ndarray:
    """Mix each channel with its spatial neighbours (volume-conduction cross-talk)."""
    x = _check_windows(x)
    rng = _rng(rng)
    n, _, C = x.shape
    alpha = rng.uniform(0.0, max_alpha, size=(n, 1, 1)).astype(np.float32)
    left = np.concatenate([x[:, :, :1], x[:, :, :-1]], axis=2)
    right = np.concatenate([x[:, :, 1:], x[:, :, -1:]], axis=2)
    return x + alpha * 0.5 * (left + right)


def channel_dropout(x, p: float = 0.05, rng=None) -> np.ndarray:
    """Zero out whole channels at random (simulates a loose or failed electrode)."""
    x = _check_windows(x)
    rng = _rng(rng)
    mask = rng.random((x.shape[0], 1, x.shape[2])) >= p
    return x * mask.astype(np.float32)


def baseline_wander(x, fs: float, amp: float = 0.05, rng=None) -> np.ndarray:
    """Add a slow (0.2-2 Hz) sinusoidal drift scaled to each channel's RMS."""
    x = _check_windows(x)
    rng = _rng(rng)
    n, T, C = x.shape
    t = np.arange(T) / fs
    freq = rng.uniform(0.2, 2.0, size=(n, 1, C))
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, C))
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True)) + 1e-12
    drift = np.sin(2 * np.pi * freq * t[None, :, None] + phase)
    return x + (amp * rms * drift).astype(np.float32)


def powerline(x, fs: float, mains_hz: float = 50.0, amp: float = 0.05, rng=None) -> np.ndarray:
    """Add mains interference (50 or 60 Hz) with random phase, scaled to channel RMS."""
    x = _check_windows(x)
    rng = _rng(rng)
    n, T, C = x.shape
    t = np.arange(T) / fs
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, C))
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True)) + 1e-12
    hum = np.sin(2 * np.pi * mains_hz * t[None, :, None] + phase)
    return x + (amp * rms * hum).astype(np.float32)


def mixup_same_class(x, y, alpha: float = 0.2, rng=None):
    """Convex combination of two windows from the same class (label is unchanged).

    Returns ``(x_mixed, y_mixed)`` with the same number of windows as the input.
    Classes with a single window are returned unchanged.
    """
    x = _check_windows(x)
    y = np.asarray(y)
    rng = _rng(rng)
    out = x.copy()
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if idx.size < 2:
            continue
        partner = rng.permutation(idx)
        lam = rng.beta(alpha, alpha, size=(idx.size, 1, 1)).astype(np.float32)
        out[idx] = lam * x[idx] + (1 - lam) * x[partner]
    return out, y.copy()


TRANSFORMS = {
    "noise": add_gaussian_noise,
    "amplitude": random_amplitude_scale,
    "time_warp": time_warp,
    "electrode_shift": electrode_shift,
    "crosstalk": crosstalk,
    "channel_dropout": channel_dropout,
    "baseline_wander": baseline_wander,
    "powerline": powerline,
}


def augment_dataset(
    x,
    y,
    n_copies: int = 1,
    ops: tuple[str, ...] = ("noise", "amplitude", "time_warp", "electrode_shift", "crosstalk"),
    fs: float = 2000.0,
    mixup: bool = True,
    rng=None,
):
    """Return originals plus ``n_copies`` augmented copies of every window.

    Each copy applies every operation in ``ops`` in order, with its own random
    parameters, so copies differ from each other and from the originals.
    Labels are duplicated unchanged.
    """
    x = _check_windows(x)
    y = np.asarray(y)
    rng = _rng(rng)
    unknown = set(ops) - set(TRANSFORMS)
    if unknown:
        raise ValueError(f"unknown augmentation ops: {sorted(unknown)}")

    xs, ys = [x], [y]
    for _ in range(n_copies):
        xa = x.copy()
        for op in ops:
            fn = TRANSFORMS[op]
            if op == "baseline_wander" or op == "powerline":
                xa = fn(xa, fs=fs, rng=rng)
            else:
                xa = fn(xa, rng=rng)
        if mixup:
            xa, _ = mixup_same_class(xa, y, rng=rng)
        xs.append(xa)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
