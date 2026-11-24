"""Physiology-inspired synthetic sEMG generator.

A deliberately small model in the spirit of the simulators in
``ime-luebeck/semgsim`` (Petersen & Rostalski 2019) and ``Josef-Djarf/sEMG-Sim``:

1. A pool of motor units (MUs) is recruited with the size principle: MU ``m``
   becomes active once the class activation level exceeds its threshold, and
   larger MUs have larger action potentials.
2. Each active MU fires a renewal spike train whose rate grows with activation.
3. Each MU's spike train is convolved with a biphasic motor-unit action potential (MUAP).
4. MU activity is projected onto the electrode array with a class-specific
   spatial gain matrix (a distinct muscle-activation pattern per gesture).
5. Per-window variability (gain jitter, recruitment jitter) and noise
   (white noise, mains hum) are added.

This is NOT a validated biophysical model. Use it to create extra training
windows for gesture classes, and check the output against real recordings
before trusting it for anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class SyntheticConfig:
    fs: float = 2000.0           # sampling rate (Hz)
    window_ms: float = 150.0     # window length (ms)
    n_channels: int = 8          # electrodes
    n_motor_units: int = 40      # MU pool size
    muap_ms: tuple[float, float] = (2.0, 8.0)  # MUAP duration range (ms), small MUs -> longer
    min_rate_hz: float = 8.0
    max_rate_hz: float = 35.0
    isi_cv: float = 0.2          # irregularity of firing (coefficient of variation of ISI)
    spatial_decay: float = 2.0   # electrode-count length scale for MU -> channel gain
    synergy_jitter: float = 0.3  # class-specific deviation of the spatial pattern
    subject_jitter: float = 0.1  # per-window log-gain jitter
    level_jitter: float = 0.03   # per-window activation jitter
    noise_std: float = 0.02      # white noise (relative to the peak MUAP amplitude)
    mains_hz: float = 50.0
    mains_amp: float = 0.02
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class ClassModel:
    """Parameters that define one gesture class."""

    level: float          # activation level in (0, 1]
    gains: np.ndarray     # (n_channels, n_motor_units) spatial pattern


def _muap_template(dur_ms: float, fs: float) -> np.ndarray:
    """Biphasic MUAP: first derivative of a Gaussian, peak-normalised to 1."""
    n = max(int(round(dur_ms * fs / 1000.0)), 3)
    t = np.linspace(-1, 1, n)
    sigma = 0.35
    g = -t * np.exp(-(t**2) / (2 * sigma**2))
    return (g / np.max(np.abs(g))).astype(np.float64)


def make_class_models(n_classes: int, cfg: SyntheticConfig, rng: np.random.Generator) -> list[ClassModel]:
    """Create activation levels and spatial patterns for each class.

    Levels are spread over (0.15, 0.95) so that classes differ in force. Each class
    gets its own spatial pattern: a distance-decay base gain plus a random synergy
    perturbation.
    """
    C, M = cfg.n_channels, cfg.n_motor_units
    levels = np.linspace(0.15, 0.95, n_classes)
    levels = np.clip(levels + rng.normal(0, 0.02, size=n_classes), 0.05, 1.0)
    mu_pos = rng.uniform(0, C - 1, size=M)                 # MU "fibre" position across array
    ch_pos = np.arange(C)
    base = np.exp(-np.abs(ch_pos[:, None] - mu_pos[None, :]) / cfg.spatial_decay)
    models = []
    for lvl in levels:
        synergy = 1.0 + cfg.synergy_jitter * rng.standard_normal((C, M))
        gains = base * np.clip(synergy, 0.1, None)
        models.append(ClassModel(level=float(lvl), gains=gains))
    return models


def _spike_train(rate_hz: float, n: int, fs: float, cv: float, rng: np.random.Generator) -> np.ndarray:
    """Renewal spike train (gamma ISIs) as a binary vector of length n."""
    spikes = np.zeros(n)
    if rate_hz <= 0:
        return spikes
    mean_isi = fs / rate_hz
    shape = 1.0 / max(cv, 1e-3) ** 2
    scale = mean_isi / shape
    t = rng.uniform(0, mean_isi)
    while t < n:
        spikes[int(t)] += 1.0
        t += rng.gamma(shape, scale)
    return spikes


def synthesize_window(model: ClassModel, cfg: SyntheticConfig, rng: np.random.Generator,
                      muap_bank: list[np.ndarray], amps: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Generate one window of shape (n_samples, n_channels)."""
    T = int(round(cfg.window_ms * cfg.fs / 1000.0))
    C, M = cfg.n_channels, cfg.n_motor_units

    level = np.clip(model.level + rng.normal(0, cfg.level_jitter), 0.0, 1.0)
    active = level >= thresholds
    # Firing rate rises with the excess of level over each MU's threshold
    excess = np.clip((level - thresholds) / np.maximum(1 - thresholds, 1e-6), 0, 1)
    rates = cfg.min_rate_hz + (cfg.max_rate_hz - cfg.min_rate_hz) * excess

    mu_signal = np.zeros((T, M))
    for m in np.flatnonzero(active):
        spikes = _spike_train(rates[m], T, cfg.fs, cfg.isi_cv, rng)
        mu_signal[:, m] = fftconvolve(spikes, muap_bank[m] * amps[m], mode="full")[:T]

    gain = model.gains * np.exp(rng.normal(0, cfg.subject_jitter, size=(C, 1)))
    window = mu_signal @ gain.T                                 # (T, C)

    t = np.arange(T) / cfg.fs
    peak = np.max(np.abs(window)) + 1e-9
    window += rng.standard_normal((T, C)) * cfg.noise_std * peak
    phase = rng.uniform(0, 2 * np.pi, size=(1, C))
    window += cfg.mains_amp * peak * np.sin(2 * np.pi * cfg.mains_hz * t[:, None] + phase)
    return window


def generate_synthetic_semg(
    n_classes: int,
    windows_per_class: int,
    cfg: SyntheticConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ClassModel]]:
    """Generate an artificial sEMG dataset.

    Returns:
        X: float32 array (n_classes * windows_per_class, n_samples, n_channels)
        y: int64 labels (0 .. n_classes - 1)
        models: the class models used (levels and spatial patterns), for reproducibility.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    models = make_class_models(n_classes, cfg, rng)

    M = cfg.n_motor_units
    lo, hi = cfg.muap_ms
    dur = np.linspace(hi, lo, M)                    # large MUs -> short MUAPs
    muap_bank = [_muap_template(d, cfg.fs) for d in dur]
    amps = 0.2 * np.exp(2.0 * np.arange(M) / M)     # size principle: amplitude grows with index
    thresholds = (np.arange(M) + 1) / M * 0.9       # recruitment thresholds, low -> high

    X, y = [], []
    for c, model in enumerate(models):
        for _ in range(windows_per_class):
            X.append(synthesize_window(model, cfg, rng, muap_bank, amps, thresholds))
            y.append(c)
    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], models
