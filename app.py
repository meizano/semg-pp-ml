r"""
sEMG Preprocessing and Machine Learning Platform for NinaPro DB8
Main Streamlit Application with Bilingual Support & Dataset Inspector
"""
import sys
import os
from pathlib import Path
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
from semg_pp_ml.utils.system import get_system_info
from semg_pp_ml.config import load_config
from semg_pp_ml.utils.dataset_checker import inspect_ninapro_dataset

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="sEMG Preprocessing & ML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Localization (Language Switcher)
# -----------------------------------------------------------------------------
locales_dir = Path(__file__).parent / "config" / "locales"
lang_choice = st.sidebar.selectbox("🌐 Language / Bahasa", ["English", "Bahasa Indonesia"], index=0)
lang_file = locales_dir / ("lang_eng.json" if lang_choice == "English" else "lang_id.json")

with open(lang_file, "r", encoding="utf-8") as f:
    t = json.load(f)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Dashboard
# -----------------------------------------------------------------------------
st.markdown(f'<div class="main-header">⚡ {t["app_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Modular research, preprocessing, feature extraction, model zoo, and TinyML pipeline for NinaPro DB8</div>', unsafe_allow_html=True)

# System Info Sidebar / Summary
sys_info = get_system_info()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">💻 {sys_info.cpu_cores} Cores</div>
        <div class="metric-label">CPU: {sys_info.cpu_name[:20]}...</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">🧠 {sys_info.ram_gb} GB</div>
        <div class="metric-label">System RAM</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">🎮 {sys_info.gpu_name if sys_info.gpu_available else 'CPU Mode'}</div>
        <div class="metric-label">Acceleration</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">🌐 {lang_choice}</div>
        <div class="metric-label">Active Locale</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# Dataset Checker Summary Section
# -----------------------------------------------------------------------------
st.subheader(t["dataset_checker_title"])
data_path = Path("data/ninapro_db8")
with st.spinner("Inspecting NinaPro DB8 dataset..."):
    ds_report = inspect_ninapro_dataset(data_path)

col_d1, col_d2 = st.columns(2)
with col_d1:
    st.info(f"📂 **Dataset Directory**: `{ds_report['data_dir']}`\n\n"
            f"• Found Subjects: `{ds_report['subjects_found']}`\n"
            f"• Total .mat Files: `{ds_report['total_files']}`\n"
            f"• EMG Data Loaded: `{'Yes' if ds_report['has_emg'] else 'No'}`")
with col_d2:
    st.success(f"🎯 **Paradigm Guide**:\n\n"
               f"• **Classification**: {ds_report['paradigm_guide']['classification']}\n\n"
               f"• **Regression**: {ds_report['paradigm_guide']['regression']}")

st.divider()

# Workflow Navigation Cards
st.markdown("### 🧭 Workflow Navigation")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    ### ⚡ 1. Preprocessing Lab
    - Granular filter configuration (Butterworth, VMD, Wavelet, Kalman).
    - SNR, PSD band power, and RMSE residual evaluation.
    - Export preprocessed `.npz` datasets.
    """)
    if st.button("Go to Preprocessing Lab ➡️", key="btn_prep", use_container_width=True):
        st.switch_page("pages/1_⚡_Preprocessing_Lab.py")

with col_b:
    st.markdown("""
    ### 🧠 2. Model Training Lab & Benchmarking
    - Model Zoo (25+ Classical, DL, Transformer, Graph, & TinyML models).
    - Step-by-step epoch CSV logging with datetime stamps.
    - Real-time Performance Benchmarking (WPS, Latency, CPU %, RAM MB).
    """)
    if st.button("Go to Model Training Lab ➡️", key="btn_model", use_container_width=True):
        st.switch_page("pages/2_🧠_Model_Training.py")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Local & LAN Mode**: Access this app anywhere on your local network.")
st.sidebar.caption("NinaPro DB8 sEMG Framework v0.1.0")
