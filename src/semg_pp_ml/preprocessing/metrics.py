"""
Preprocessing quality metrics for sEMG signals.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal
from typing import Dict, Any


def compute_snr(signal_clean: np.ndarray, signal_raw: np.ndarray) -> Dict[str, Any]:
    """
    Compute Signal-to-Noise Ratio (SNR) in dB per channel and overall.

    Args:
        signal_clean: Filtered/processed signal (n_samples, n_channels)
        signal_raw: Original raw signal (n_samples, n_channels)

    Returns:
        Dict with per-channel and aggregate SNR metrics
    """
    noise = signal_raw - signal_clean
    signal_power = np.mean(signal_clean ** 2, axis=0) + 1e-12
    noise_power = np.mean(noise ** 2, axis=0) + 1e-12
    snr_db = 10.0 * np.log10(signal_power / noise_power)

    return {
        'snr_per_channel_db': snr_db.tolist(),
        'snr_mean_db': float(np.mean(snr_db)),
        'snr_std_db': float(np.std(snr_db)),
        'snr_min_db': float(np.min(snr_db)),
        'snr_max_db': float(np.max(snr_db)),
    }


def compute_psd_metrics(
    data: np.ndarray,
    fs: int,
    fmin: float = 10.0,
    fmax: float = 500.0
) -> Dict[str, Any]:
    """
    Compute Power Spectral Density (PSD) metrics.

    Args:
        data: Signal (n_samples, n_channels)
        fs: Sampling frequency
        fmin: Minimum frequency for band power
        fmax: Maximum frequency for band power

    Returns:
        Dict with band power, total power, and ratios
    """
    n_ch = data.shape[1]
    band_powers = []
    total_powers = []
    freqs_list = None

    for ch in range(n_ch):
        nperseg = min(1024, data.shape[0] // 4)
        if nperseg < 8:
            nperseg = data.shape[0]
        f, Pxx = signal.welch(data[:, ch], fs, nperseg=nperseg)

        if freqs_list is None:
            freqs_list = f.tolist()

        total = np.trapezoid(Pxx, f) + 1e-12
        idx = (f >= fmin) & (f <= fmax)
        band = np.trapezoid(Pxx[idx], f[idx]) if np.any(idx) else 0.0

        band_powers.append(float(band))
        total_powers.append(float(total))

    ratios = [b / t for b, t in zip(band_powers, total_powers)]

    return {
        'psd_band_power': band_powers,
        'psd_total_power': total_powers,
        'psd_band_ratio': ratios,
        'psd_mean_ratio': float(np.mean(ratios)),
        'psd_frequencies': freqs_list,
        'psd_fmin': fmin,
        'psd_fmax': fmax,
    }


def compute_preprocessing_rmse(
    signal_raw: np.ndarray,
    signal_filtered: np.ndarray
) -> Dict[str, Any]:
    """
    Compute RMSE and MAE of noise residual (raw - filtered).

    Args:
        signal_raw: Original signal
        signal_filtered: Filtered signal

    Returns:
        Dict with per-channel and aggregate RMSE/MAE
    """
    noise = signal_raw - signal_filtered
    rmse_per_ch = np.sqrt(np.mean(noise ** 2, axis=0))
    mae_per_ch = np.mean(np.abs(noise), axis=0)

    return {
        'rmse_noise_per_channel': rmse_per_ch.tolist(),
        'rmse_noise_mean': float(np.mean(rmse_per_ch)),
        'rmse_noise_std': float(np.std(rmse_per_ch)),
        'mae_noise_per_channel': mae_per_ch.tolist(),
        'mae_noise_mean': float(np.mean(mae_per_ch)),
        'mae_noise_std': float(np.std(mae_per_ch)),
    }


def compute_all_preprocessing_metrics(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: int
) -> Dict[str, Any]:
    """Compute all preprocessing metrics at once."""
    metrics = {}
    metrics.update(compute_snr(filtered, raw))
    metrics.update(compute_psd_metrics(filtered, fs))
    metrics.update(compute_preprocessing_rmse(raw, filtered))
    return metrics