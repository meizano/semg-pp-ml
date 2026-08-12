r"""
Independent Model Performance Evaluation Page
Measure accuracy, latency, WPS, CPU, and RAM usage of saved trained models.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib
import plotly.express as px

from semg_pp_ml.models.registry import load_model, is_classical_model
from semg_pp_ml.metrics.classification import compute_all_classification_metrics
from semg_pp_ml.utils.benchmarking import benchmark_model_inference

st.set_page_config(page_title="Model Performance - sEMG Platform", page_icon="⚡", layout="wide")

st.title("⚡ Independent Model Performance Evaluation")
st.caption("Load any saved model (`.pt` or `.pkl`) and test dataset (`.npz`) to evaluate classification metrics, inference latency, WPS, and resource usage.")

# -----------------------------------------------------------------------------
# 1. Select Model & Dataset
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Select Saved Model")
    model_dir = Path("outputs/trained_models")
    model_files = sorted(list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pkl"))) if model_dir.exists() else []

    if not model_files:
        st.warning("No saved models found in `outputs/trained_models`.")
        st.stop()

    selected_model_path = st.selectbox("Choose Model", model_files, format_func=lambda p: p.name)

with col2:
    st.subheader("2. Select Test Dataset (.npz)")
    prep_dir = Path("outputs/preprocessed_data")
    npz_files = sorted(list(prep_dir.glob("*.npz"))) if prep_dir.exists() else []

    if not npz_files:
        st.warning("No preprocessed datasets found in `outputs/preprocessed_data`.")
        st.stop()

    selected_npz_path = st.selectbox("Choose Dataset", npz_files, format_func=lambda p: p.name)

# -----------------------------------------------------------------------------
# 3. Evaluate Performance Button
# -----------------------------------------------------------------------------
if st.button("🚀 Evaluate Model Performance", type="primary", use_container_width=True):
    with st.spinner("Loading model, dataset, and running performance benchmark..."):
        try:
            # Load Dataset
            data = np.load(selected_npz_path)
            X_te = data['X_te'] if 'X_te' in data else data['emg_scaled'][:1000]
            y_te = data['y_te'] if 'y_te' in data else data.get('labels', np.zeros(len(X_te)))

            # Determine model type
            stem = selected_model_path.stem
            model_type = stem.split('_')[1] if len(stem.split('_')) > 1 else ('svm' if selected_model_path.suffix == '.pkl' else 'cnn1d')

            # Load Model
            if selected_model_path.suffix == '.pkl':
                model = joblib.load(selected_model_path)
            else:
                model = load_model(selected_model_path, model_type)

            # Benchmark Inference
            bench = benchmark_model_inference(model, X_te)

            # Predictions
            if is_classical_model(model_type):
                X_te_2d = X_te.reshape(X_te.shape[0], -1)
                y_prob = model.predict_proba(X_te_2d)
            else:
                y_prob = model.predict_proba(X_te)

            y_pred = np.argmax(y_prob, axis=1)
            y_true = np.argmax(y_te, axis=1) if y_te.ndim > 1 else y_te.astype(int)
            n_classes = len(np.unique(y_true))

            metrics = compute_all_classification_metrics(
                y_true, y_pred, y_prob, y_te if y_te.ndim > 1 else np.eye(n_classes)[y_true],
                n_classes, [f"Class-{c}" for c in range(n_classes)]
            )

            st.success("🎉 Performance Evaluation Complete!")

            # Display Benchmark Metrics
            st.markdown("### ⚡ Benchmark Metrics")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Windows Per Sec (WPS)", f"{bench['windows_per_second_wps']} win/s")
            b2.metric("Inference Latency", f"{bench['latency_ms_per_window']} ms")
            b3.metric("CPU Utilization", f"{bench['cpu_usage_percent']}%")
            b4.metric("RAM Footprint", f"{bench['ram_footprint_mb']} MB")

            st.divider()

            # Display Classification Metrics
            st.markdown("### 📊 Classification Accuracy Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
            m2.metric("F1-Score (Weighted)", f"{metrics.get('f1_score', 0) * 100:.2f}%")
            m3.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
            m4.metric("Recall", f"{metrics.get('recall', 0) * 100:.2f}%")

            # Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            import traceback
            st.code(traceback.format_exc())
