"""
Signal visualization for sEMG preprocessing.
"""
from __future__ import annotations

import os
import numpy as np
import scipy.signal as signal
from pathlib import Path
from typing import List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_signal_comparison(
    emg_raw: np.ndarray,
    emg_filtered: np.ndarray,
    fs: int,
    channels: List[int],
    output_dir: str | Path,
    prep_name: str,
    n_plot_samples: int = 2000,
    title: Optional[str] = None
) -> str:
    """Plot raw vs filtered signal comparison."""
    if not HAS_MATPLOTLIB:
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_plot = min(n_plot_samples, emg_raw.shape[0])
    t = np.arange(n_plot) / fs
    n_ch = len(channels)

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for idx, ch_idx in enumerate(channels):
        ax = axes[idx]
        ax.plot(t, emg_raw[:n_plot, idx], alpha=0.5, color='gray', label='Raw', linewidth=0.8)
        ax.plot(t, emg_filtered[:n_plot, idx], color='red', label='Filtered', linewidth=1.0)
        ax.set_title(f'Channel {ch_idx + 1} - {prep_name}', fontweight='bold', fontsize=10)
        ax.set_ylabel('Amplitude (mV)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (seconds)')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    fname = output_dir / f"signal_comparison_{prep_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return str(fname)


def plot_preprocessing_metrics(
    snr_per_ch: List[float],
    psd_ratios: List[float],
    rmse_per_ch: List[float],
    mae_per_ch: List[float],
    channel_indices: List[int],
    output_dir: str | Path,
    prep_name: str,
    snr_mean: float,
    psd_mean: float,
    rmse_mean: float,
    mae_mean: float
) -> str:
    """Plot preprocessing quality metrics."""
    if not HAS_MATPLOTLIB:
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ch_labels = [f'Ch{i+1}' for i in channel_indices]

    # SNR
    ax = axes[0, 0]
    bars = ax.bar(ch_labels, snr_per_ch, edgecolor='black', linewidth=0.5, color='#e74c3c')
    ax.axhline(snr_mean, color='green', linestyle='--', linewidth=2, label=f'Mean={snr_mean:.2f} dB')
    ax.set_title('SNR per Channel', fontweight='bold')
    ax.set_ylabel('SNR (dB)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # PSD Ratio
    ax = axes[0, 1]
    bars = ax.bar(ch_labels, psd_ratios, edgecolor='black', linewidth=0.5, color='#3498db')
    ax.axhline(psd_mean, color='green', linestyle='--', linewidth=2, label=f'Mean={psd_mean:.4f}')
    ax.set_title('PSD Ratio (10-500 Hz) per Channel', fontweight='bold')
    ax.set_ylabel('Band / Total Power')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # RMSE
    ax = axes[1, 0]
    bars = ax.bar(ch_labels, rmse_per_ch, edgecolor='black', linewidth=0.5, color='#f39c12')
    ax.axhline(rmse_mean, color='green', linestyle='--', linewidth=2, label=f'Mean={rmse_mean:.6f}')
    ax.set_title('RMSE Noise Residual per Channel', fontweight='bold')
    ax.set_ylabel('RMSE (mV)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # MAE
    ax = axes[1, 1]
    bars = ax.bar(ch_labels, mae_per_ch, edgecolor='black', linewidth=0.5, color='#9b59b6')
    ax.axhline(mae_mean, color='green', linestyle='--', linewidth=2, label=f'Mean={mae_mean:.6f}')
    ax.set_title('MAE Noise Residual per Channel', fontweight='bold')
    ax.set_ylabel('MAE (mV)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fname = output_dir / f"preprocessing_metrics_{prep_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return str(fname)


def plot_psd_comparison(
    emg_raw: np.ndarray,
    emg_filtered: np.ndarray,
    fs: int,
    channels: List[int],
    output_dir: str | Path,
    prep_name: str
) -> str:
    """Plot PSD comparison between raw and filtered signals."""
    if not HAS_MATPLOTLIB:
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_ch = len(channels)
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for idx, ch_idx in enumerate(channels):
        ax = axes[idx]

        nperseg = min(1024, emg_raw.shape[0] // 4)
        f_raw, Pxx_raw = signal.welch(emg_raw[:, ch_idx], fs, nperseg=nperseg)
        f_filt, Pxx_filt = signal.welch(emg_filtered[:, ch_idx], fs, nperseg=nperseg)

        ax.semilogy(f_raw, Pxx_raw, alpha=0.5, color='gray', label='Raw', linewidth=0.8)
        ax.semilogy(f_filt, Pxx_filt, color='red', label='Filtered', linewidth=1.0)
        ax.set_title(f'Channel {ch_idx + 1} PSD - {prep_name}', fontweight='bold')
        ax.set_ylabel('Power (dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, fs / 2)

    axes[-1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    fname = output_dir / f"psd_comparison_{prep_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return str(fname)