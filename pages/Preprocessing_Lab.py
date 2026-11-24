r"""
Preprocessing Lab - Standalone Signal Processing, Filtering, Metrics, and Export
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import numpy as np
import plotly.graph_objects as me_plotly
from plotly.subplots import make_subplots
import scipy.signal as signal
import json
import joblib
import pandas as pd

from semg_pp_ml.config import ExperimentConfig, PreprocessingConfig, ButterworthConfig, KalmanConfig, VMDConfig, WaveletConfig, NotchConfig
from semg_pp_ml.data.loader import load_mat_files_from_folder
from semg_pp_ml.preprocessing.filters import apply_preprocessing
from semg_pp_ml.preprocessing.metrics import compute_all_preprocessing_metrics

st.set_page_config(page_title="Preprocessing Lab - sEMG Platform", page_icon="⚡", layout="wide")

st.title("Preprocessing Lab (Standalone)")
st.caption("Configure granular filtering parameters, evaluate SNR / PSD / RMSE metrics, visualize interactive signals, and save preprocessed datasets for later ML experiments.")

# -----------------------------------------------------------------------------
# Configuration Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Dataset & Signal Settings")

data_dir = st.sidebar.text_input("Train Data Folder", "data/ninapro_db8")
subjects = st.sidebar.multiselect("Subjects", list(range(1, 11)), default=[1])
channels = st.sidebar.multiselect("Channels (0-indexed)", list(range(16)), default=[3, 4, 5, 11, 12, 13])
fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=2000, min_value=100, max_value=10000, step=100)

st.sidebar.header("Preprocessing Method")
prep_method = st.sidebar.selectbox(
    "Filter Algorithm",
    ["butterworth_kalman", "butterworth", "kalman", "vmd", "wavelet", "notch", "raw"],
    format_func=lambda x: {
        "raw": "1. Raw (No Filter)",
        "butterworth": "2. Butterworth Bandpass",
        "kalman": "3. Kalman Filter",
        "butterworth_kalman": "4. Butterworth + Kalman",
        "vmd": "5. Variational Mode Decomposition (VMD)",
        "wavelet": "6. Wavelet Denoising",
        "notch": "7. Notch Filter (50/60 Hz)"
    }[x]
)

prep_cfg = PreprocessingConfig(method=prep_method)

with st.sidebar.expander("Filter Hyperparameters", expanded=True):
    if "butterworth" in prep_method:
        st.markdown("**Butterworth Bandpass**")
        order = st.number_input("Order", value=4, min_value=1, max_value=8)
        low_hz = st.number_input("Low Cutoff (Hz)", value=10.0, min_value=0.1, max_value=200.0)
        high_hz = st.number_input("High Cutoff (Hz)", value=400.0, min_value=50.0, max_value=1000.0)
        prep_cfg.butterworth = ButterworthConfig(order=order, low_hz=low_hz, high_hz=high_hz)

    if "kalman" in prep_method:
        st.markdown("**Kalman Filter**")
        q = st.number_input("Process Noise (Q)", value=1e-5, format="%.2e")
        r = st.number_input("Measurement Noise (R)", value=0.05, format="%.3f")
        prep_cfg.kalman = KalmanConfig(q=q, r=r)

    if prep_method == "vmd":
        st.markdown("**VMD Decomposition**")
        alpha = st.number_input("Alpha (Bandwidth)", value=2000, min_value=100, max_value=10000)
        k = st.number_input("K (Num Modes)", value=3, min_value=1, max_value=8)
        prep_cfg.vmd = VMDConfig(alpha=alpha, k=k)

    if prep_method == "wavelet":
        st.markdown("**Wavelet Denoising**")
        wavelet = st.selectbox("Wavelet Family", ["db4", "db8", "sym4", "coif3"])
        level = st.number_input("Level", value=4, min_value=1, max_value=8)
        mode = st.selectbox("Threshold Mode", ["soft", "hard"])
        prep_cfg.wavelet = WaveletConfig(wavelet=wavelet, level=level, threshold_mode=mode)

    if prep_method == "notch":
        st.markdown("**Notch Filter**")
        freq = st.selectbox("Notch Frequency (Hz)", [50.0, 60.0])
        quality = st.number_input("Quality Factor Q", value=30.0, min_value=5.0, max_value=100.0)
        prep_cfg.notch = NotchConfig(freq=freq, quality_factor=quality)

out_dir = st.sidebar.text_input("Output Directory", r"outputs\preprocessed_data")

# -----------------------------------------------------------------------------
# Main Content & Processing
# -----------------------------------------------------------------------------
if not channels:
    st.warning("Please select at least one sEMG channel in the sidebar.")
    st.stop()

if st.button("Process Signal & Compute Metrics", type="primary"):
    with st.spinner("Loading .mat dataset and applying preprocessing pipeline..."):
        try:
            emg, labels, glove = load_mat_files_from_folder(data_dir, subjects, channels)

            if emg.size == 0:
                st.error(f"No valid EMG data found in folder `{data_dir}` for subjects `{subjects}`.")
                st.stop()

            # Limit samples for interactive preview if too large
            preview_samples = min(10000, emg.shape[0])
            emg_preview = emg[:preview_samples]
            labels_preview = labels[:preview_samples] if labels.size > 0 else np.zeros(preview_samples)

            # Apply preprocessing
            res = apply_preprocessing(emg_preview, fs, prep_cfg)
            metrics = compute_all_preprocessing_metrics(emg_preview, res.signal, fs)

            st.session_state["prep_cache"] = {
                "emg_raw": emg_preview,
                "emg_filtered": res.signal,
                "labels": labels_preview,
                "fs": fs,
                "metrics": metrics,
                "prep_cfg": prep_cfg,
                "subjects": subjects,
                "channels": channels,
                "full_emg": emg,
                "full_labels": labels,
                "glove": glove,
            }
            st.success("Preprocessing pipeline executed successfully!")

        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# Display Results if cached
if "prep_cache" in st.session_state:
    cache = st.session_state["prep_cache"]
    metrics = cache["metrics"]

    st.markdown("### Signal Quality Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SNR (dB)", f"{metrics.get('snr_db', 0):.2f}")
    col2.metric("PSD Band Ratio", f"{metrics.get('psd_band_power_ratio', 0) * 100:.1f}%")
    col3.metric("RMSE Residual", f"{metrics.get('rmse_noise_residual', 0):.4f}")
    col4.metric("Samples", f"{len(cache['emg_filtered'])}")

    tab_plot, tab_psd, tab_save = st.tabs([
        "Waveform Plots",
        "PSD Periodogram",
        "Save Preprocessed Dataset"
    ])

    with tab_plot:
        st.markdown("#### Raw vs Filtered sEMG Waveform")
        ch_idx = st.selectbox("Select Channel to Visualize", list(range(len(cache["channels"]))), format_func=lambda x: f"Channel {cache['channels'][x]}")
        
        t_axis = np.arange(len(cache["emg_raw"])) / cache["fs"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Raw sEMG", "Filtered sEMG"))
        fig.add_trace(go.Scatter(x=t_axis, y=cache["emg_raw"][:, ch_idx], name="Raw", line=dict(color="#94A3B8", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_axis, y=cache["emg_filtered"][:, ch_idx], name="Filtered", line=dict(color="#2563EB", width=1.2)), row=2, col=1)
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab_psd:
        st.markdown("#### Power Spectral Density (PSD)")
        f_raw, pxx_raw = signal.welch(cache["emg_raw"][:, ch_idx], cache["fs"], nperseg=512)
        f_filt, pxx_filt = signal.welch(cache["emg_filtered"][:, ch_idx], cache["fs"], nperseg=512)

        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=f_raw, y=10 * np.log10(pxx_raw + 1e-12), name="Raw PSD", line=dict(color="#94A3B8")))
        fig_psd.add_trace(go.Scatter(x=f_psd, y=10 * np.log10(pxx_filt + 1e-12), name="Filtered PSD", line=dict(color="#16A34A")))
        fig_psd.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB/Hz)", height=400)
        st.plotly_chart(fig_psd, use_container_width=True)

    with tab_save:
        st.markdown("#### Save Preprocessed Dataset for Machine Learning")
        save_name = st.text_input("Dataset File Base Name", f"dataset_{prep_method}_S{'_'.join(map(str, subjects))}")

        if st.button("Save Preprocessed Dataset Now", type="primary"):
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            npz_file = out_path / f"{save_name}.npz"
            scaler_file = out_path / f"{save_name}_scaler.pkl"
            metrics_file = out_path / f"{save_name}_metrics.json"

            # Run full pipeline on full data
            full_res = apply_preprocessing(cache["full_emg"], cache["fs"], cache["prep_cfg"])

            np.savez_compressed(
                npz_file,
                emg_scaled=full_res.signal,
                labels=cache["full_labels"],
                glove=cache["glove"] if cache["glove"] is not None else np.array([]),
                method=prep_method,
                channels=np.array(cache["channels"]),
                fs=cache["fs"]
            )

            with open(metrics_file, "w") as jf:
                json.dumps(metrics, indent=2)

            st.success(f"Dataset saved successfully!")
            st.code(f"NPZ Path : {npz_file.resolve()}\nMetrics  : {metrics_file.resolve()}")
