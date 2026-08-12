"""
Preprocessing filters for sEMG signals.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal
from typing import Protocol
from dataclasses import dataclass

from ..config import PreprocessingConfig, ButterworthConfig, KalmanConfig, VMDConfig, WaveletConfig, NotchConfig


class KalmanFilter1D:
    """Scalar Kalman filter for 1D signal denoising."""
    def __init__(self, q: float = 1e-5, r: float = 0.05):
        self.q = q  # Process noise covariance
        self.r = r  # Measurement noise covariance
        self.x = 0.0  # State estimate
        self.p = 1.0  # Estimate error covariance
        self.k = 0.0  # Kalman gain

    def reset(self, initial_value: float = 0.0):
        self.x = float(initial_value)
        self.p = 1.0

    def update(self, measurement: float) -> float:
        # Predict
        self.p = self.p + self.q
        # Update
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1.0 - self.k) * self.p
        return self.x


class PreprocessingFilter(Protocol):
    """Protocol for preprocessing filters."""
    def __call__(self, emg: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray: ...


def raw_filter(emg: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray:
    """No filtering - return raw signal."""
    return emg.copy()


def butterworth_filter(emg: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray:
    """Bandpass Butterworth filter with zero-phase distortion (filtfilt)."""
    cfg: ButterworthConfig = config.butterworth
    nyq = 0.5 * fs
    low = cfg.low_hz / nyq
    high = cfg.high_hz / nyq

    # Ensure valid frequency range
    low = max(low, 1e-6)
    high = min(high, 1.0 - 1e-6)

    if low >= high:
        raise ValueError(f"Invalid frequency range: low={cfg.low_hz}Hz, high={cfg.high_hz}Hz at fs={fs}Hz")

    b, a = signal.butter(cfg.order, [low, high], btype='band')
    return signal.filtfilt(b, a, emg, axis=0)


def kalman_filter(emg: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray:
    """Per-channel Kalman filtering."""
    cfg: KalmanConfig = config.kalman
    n_samples, n_channels = emg.shape
    filtered = np.zeros_like(emg, dtype=np.float64)

    for c in range(n_channels):
        kf = KalmanFilter1D(q=cfg.q, r=cfg.r)
        kf.reset(emg[0, c])
        for i in range(n_samples):
            filtered[i, c] = kf.update(emg[i, c])

    return filtered


def butterworth_kalman_filter(emg: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray:
    """Sequential Butterworth + Kalman filtering."""
    temp = butterworth_filter(emg, fs, config)
    return kalman_filter(temp, fs, config)


def vmd_filter(emg: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray:
    """
    Variational Mode Decomposition (VMD) filtering.
    Extracts the highest energy IMF (Intrinsic Mode Function).
    """
    try:
        from vmdpy import VMD
    except ImportError:
        raise ImportError("vmdpy not installed. Install with: pip install vmdpy")

    cfg: VMDConfig = config.vmd
    alpha, tau, K, DC, init, tol = cfg.alpha, cfg.tau, cfg.k, cfg.dc, cfg.init, cfg.tol

    n_samples, n_channels = emg.shape
    filtered = np.zeros_like(emg, dtype=np.float64)

    # Process in chunks to manage memory
    chunk_size = min(2000, n_samples)

    for c in range(n_channels):
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk_data = emg[i:end_idx, c]

            if len(chunk_data) <= K:
                filtered[i:end_idx, c] = chunk_data
                continue

            try:
                u, u_hat, omega = VMD(chunk_data, alpha, tau, K, DC, init, tol)
                # Use the last IMF (highest frequency/energy typically)
                imf = u[-1, :]
                actual_length = len(imf)
                filtered[i:i + actual_length, c] = imf
            except Exception as e:
                print(f"[VMD Warning] Channel {c}, chunk {i}:{end_idx} failed: {e}. Using raw.")
                filtered[i:end_idx, c] = chunk_data

    return filtered


def wavelet_filter(emg: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray:
    """Wavelet denoising using PyWavelets."""
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets not installed. Install with: pip install PyWavelets")

    cfg: WaveletConfig = config.wavelet
    n_samples, n_channels = emg.shape
    filtered = np.zeros_like(emg, dtype=np.float64)

    for c in range(n_channels):
        coeffs = pywt.wavedec(emg[:, c], cfg.wavelet, level=cfg.level)
        # Soft/hard thresholding on detail coefficients
        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(n_samples))
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            if cfg.threshold_mode == "soft":
                new_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
            else:
                new_coeffs.append(pywt.threshold(detail, threshold, mode='hard'))
        filtered[:, c] = pywt.waverec(new_coeffs, cfg.wavelet)[:n_samples]

    return filtered


def notch_filter(emg: np.ndarray, fs: int, config: PreprocessingConfig) -> np.ndarray:
    """Notch filter for powerline interference (50/60 Hz)."""
    cfg: NotchConfig = config.notch
    nyq = 0.5 * fs
    freq = cfg.freq / nyq
    b, a = signal.iirnotch(freq, cfg.quality_factor)
    return signal.filtfilt(b, a, emg, axis=0)


# Filter registry
FILTER_REGISTRY: dict[str, PreprocessingFilter] = {
    'raw': raw_filter,
    'butterworth': butterworth_filter,
    'kalman': kalman_filter,
    'butterworth_kalman': butterworth_kalman_filter,
    'vmd': vmd_filter,
    'wavelet': wavelet_filter,
    'notch': notch_filter,
}


def get_filter(method: str) -> PreprocessingFilter:
    """Get filter function by name."""
    if method not in FILTER_REGISTRY:
        raise ValueError(f"Unknown preprocessing method: {method}. Available: {list(FILTER_REGISTRY.keys())}")
    return FILTER_REGISTRY[method]


def apply_preprocessing(emg: np.ndarray, fs: int, method: str, config: PreprocessingConfig) -> np.ndarray:
    """Apply preprocessing method to EMG data."""
    filter_fn = get_filter(method)
    return filter_fn(emg, fs, config)