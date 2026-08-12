r"""
Model Training Lab - Standalone Machine Learning & Deep Learning Engine
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
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

from semg_pp_ml.config import ExperimentConfig
from semg_pp_ml.data.loader import load_mat_files_from_folder, sliding_window, train_test_split_stratified
from semg_pp_ml.preprocessing.filters import apply_preprocessing
from semg_pp_ml.models.registry import (
    MODEL_REGISTRY, MODEL_NAMES, MODEL_CATEGORIES,
    build_model, count_params, save_model, is_classical_model,
    get_default_hyperparameters, merge_hyperparameters
)
from semg_pp_ml.metrics.classification import compute_all_classification_metrics
from semg_pp_ml.utils.benchmarking import benchmark_model_inference

st.set_page_config(page_title="Model Training Lab - sEMG Platform", page_icon="⚡", layout="wide")

st.title("Model Training Lab (Standalone)")
st.caption("Load saved preprocessed datasets or run live, select model architecture, tune hyperparameters, train, evaluate metrics, and save trained models.")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source", ["Use Preprocessed Dataset (.npz)", "Run Live Preprocessing (.mat)"])

# Common settings
window_size = st.sidebar.number_input("Window Size", value=400, min_value=50, max_value=2000, step=50)
overlap = st.sidebar.number_input("Overlap", value=200, min_value=0, max_value=1900, step=25)

X_tr, X_te, y_tr, y_te, le, n_classes = None, None, None, None, None, 0

if data_source == "Use Preprocessed Dataset (.npz)":
    prep_dir = Path("outputs/preprocessed_data")
    npz_files = sorted(list(prep_dir.glob("*.npz"))) if prep_dir.exists() else []

    if not npz_files:
        st.sidebar.warning("No preprocessed datasets found. Please process data in Preprocessing Lab first.")
    else:
        selected_npz = st.sidebar.selectbox("Choose Preprocessed Dataset", npz_files, format_func=lambda p: p.name)
        prep_name = selected_npz.stem

        if st.sidebar.button("Load Dataset", type="primary"):
            with st.spinner("Loading .npz dataset and applying sliding windows..."):
                data = np.load(selected_npz)
                emg = data['emg_scaled']
                labels = data['labels']

                # Sliding window
                unique_classes = np.unique(labels)
                X, y, _ = sliding_window(emg, labels, window_size, overlap, selected_classes=list(map(int, unique_classes)))
                
                if len(X) > 0:
                    le = LabelEncoder()
                    y_enc = le.fit_transform(y)
                    n_classes = len(np.unique(y_enc))
                    
                    # One-hot encoding for deep learning
                    y_cat = np.eye(n_classes)[y_enc]

                    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y_cat, test_size=0.2, random_state=42)
                    st.session_state["dataset_cache"] = (X_tr, X_te, y_tr, y_te, le, n_classes, prep_name)
                    st.success(f"Dataset loaded! Samples: {len(X)} | Classes: {n_classes}")

if "dataset_cache" in st.session_state:
    X_tr, X_te, y_tr, y_te, le, n_classes, prep_name = st.session_state["dataset_cache"]

st.sidebar.header("Model Architecture")
model_category = st.sidebar.selectbox("Category", list(MODEL_CATEGORIES.keys()))
available_models_in_cat = MODEL_CATEGORIES[model_category]
model_type = st.sidebar.selectbox("Model Architecture", available_models_in_cat, format_func=lambda x: MODEL_NAMES.get(x, x))

st.sidebar.header("Training Hyperparameters")
lr = st.sidebar.number_input("Learning Rate", value=1e-3, format="%.1e")
epochs = st.sidebar.number_input("Epochs", value=15, min_value=1, max_value=200)
batch_size = st.sidebar.number_input("Batch Size", value=32, min_value=8, max_value=256)

# Hyperparameters
default_hp = get_default_hyperparameters(model_type)
model_params = {}
with st.sidebar.expander("Model Hyperparameters"):
    for k, v in default_hp.items():
        if isinstance(v, int):
            model_params[k] = st.number_input(k, value=v)
        elif isinstance(v, float):
            model_params[k] = st.number_input(k, value=v)
        else:
            model_params[k] = st.text_input(k, value=str(v))

out_models_dir = Path("outputs/trained_models")

# -----------------------------------------------------------------------------
# Main Training & Evaluation Panel
# -----------------------------------------------------------------------------
if X_tr is None:
    st.info("👈 Please load a dataset in the sidebar to begin model training.")
else:
    st.markdown(f"### Selected Model: **{MODEL_NAMES.get(model_type, model_type)}** ({model_category})")

    if st.button("Train & Evaluate Model", type="primary"):
        with st.spinner(f"Training {MODEL_NAMES.get(model_type, model_type)}..."):
            t0_train = time.perf_counter()

            input_shape = (window_size, X_tr.shape[2])
            merged_params = merge_hyperparameters(model_type, model_params)
            model = build_model(model_type, input_shape, n_classes, lr, merged_params)
            n_params, trainable_params, _ = count_params(model)

            if is_classical_model(model_type):
                X_tr_2d = X_tr.reshape(X_tr.shape[0], -1)
                y_tr_1d = np.argmax(y_tr, axis=1) if y_tr.ndim > 1 else y_tr
                model.fit(X_tr_2d, y_tr_1d)
                train_time = time.perf_counter() - t0_train
                history_dict = None
                epochs_run = 1
            else:
                history = model.fit(
                    X_tr, y_tr,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                train_time = time.perf_counter() - t0_train
                history_dict = history.history
                epochs_run = len(history_dict['loss'])

        # Inference & Evaluation
        with st.spinner("Evaluating on Test Set..."):
            t0_inf = time.perf_counter()
            if is_classical_model(model_type):
                X_te_2d = X_te.reshape(X_te.shape[0], -1)
                y_prob = model.predict_proba(X_te_2d)
            else:
                y_prob = model.predict_proba(X_te)

            t_inf = time.perf_counter() - t0_inf
            latency_ms = (t_inf / len(X_te)) * 1000.0

            y_pred = np.argmax(y_prob, axis=1)
            y_true = np.argmax(y_te, axis=1) if y_te.ndim > 1 else y_te.astype(int)

            class_names = [f"Class-{c}" for c in le.classes_]
            metrics = compute_all_classification_metrics(y_true, y_pred, y_prob, y_te if y_te.ndim > 1 else np.eye(n_classes)[y_true], n_classes, class_names)

            st.session_state["trained_cache"] = {
                "model": model,
                "model_type": model_type,
                "prep_name": prep_name,
                "metrics": metrics,
                "history_dict": history_dict,
                "y_true": y_true,
                "y_pred": y_pred,
                "class_names": class_names,
                "train_time": train_time,
                "latency_ms": latency_ms,
                "epochs_run": epochs_run,
                "n_params": n_params,
                "le": le,
                "X_test": X_te,
            }

    if "trained_cache" in st.session_state:
        cache = st.session_state["trained_cache"]
        metrics = cache["metrics"]
        m_type = cache["model_type"]
        prep_name = cache["prep_name"]
        history_dict = cache["history_dict"]
        y_true = cache["y_true"]
        y_pred = cache["y_pred"]
        class_names = cache["class_names"]

        st.success(f"Model trained successfully in **{cache['train_time']:.2f}s**!")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
        m2.metric("F1-Score", f"{metrics.get('f1_score', 0) * 100:.2f}%")
        m3.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
        m4.metric("Recall", f"{metrics.get('recall', 0) * 100:.2f}%")
        m5.metric("Latency", f"{cache['latency_ms']:.3f} ms/sample")

        tab_eval, tab_bench, tab_curves, tab_save = st.tabs([
            "Classification Metrics & Confusion Matrix",
            "Real-Time Performance Benchmark (WPS, CPU, RAM)",
            "Training & Validation Curves",
            "Save Trained Model Artifact"
        ])

        with tab_eval:
            col_cm, col_rep = st.columns([1, 1])
            with col_cm:
                st.markdown("#### Confusion Matrix")
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, y_pred)
                fig_cm = px.imshow(cm, x=class_names, y=class_names, text_auto=True, color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_rep:
                st.markdown("#### Detailed Classification Report")
                rep = metrics.get("classification_report", {})
                if rep:
                    df_rep = pd.DataFrame(rep).transpose()
                    st.dataframe(df_rep.style.format("{:.4f}"), use_container_width=True)

        with tab_bench:
            st.markdown("#### Real-Time Inference & System Resource Benchmarking")
            if st.button("Run Benchmark Profile Now", type="primary"):
                with st.spinner("Profiling model inference speed and resource usage..."):
                    bench = benchmark_model_inference(cache["model"], cache.get("X_test"))
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("WPS", f"{bench['windows_per_second_wps']} win/s")
                    b2.metric("Latency", f"{bench['latency_ms_per_window']} ms")
                    b3.metric("CPU", f"{bench['cpu_usage_percent']}%")
                    b4.metric("RAM", f"{bench['ram_footprint_mb']} MB")
                    st.json(bench)

        with tab_curves:
            if history_dict:
                epochs_range = list(range(1, len(history_dict['loss']) + 1))
                fig_curves = go.Figure()
                fig_curves.add_trace(go.Scatter(x=epochs_range, y=history_dict['accuracy'], mode='lines', name='Train Accuracy'))
                fig_curves.add_trace(go.Scatter(x=epochs_range, y=history_dict['val_accuracy'], mode='lines', name='Val Accuracy'))
                fig_curves.add_trace(go.Scatter(x=epochs_range, y=history_dict['loss'], mode='lines', name='Train Loss', yaxis="y2"))
                fig_curves.add_trace(go.Scatter(x=epochs_range, y=history_dict['val_loss'], mode='lines', name='Val Loss', yaxis="y2"))
                fig_curves.update_layout(xaxis_title="Epoch", yaxis=dict(title="Accuracy", range=[0, 1.05]), yaxis2=dict(title="Loss", overlaying="y", side="right"))
                st.plotly_chart(fig_curves, use_container_width=True)
            else:
                st.info("Training curves available for Deep Learning models.")

        with tab_save:
            save_model_name = st.text_input("Saved Model File Base Name", f"model_{m_type}_{prep_name}")
            if st.button("Save Model Now", type="primary"):
                out_models_dir.mkdir(parents=True, exist_ok=True)
                model_obj = cache["model"]
                if is_classical_model(m_type):
                    save_path = out_models_dir / f"{save_model_name}.pkl"
                    joblib.dump(model_obj, save_path)
                else:
                    save_path = out_models_dir / f"{save_model_name}.pt"
                    save_model(model_obj, save_path, m_type)

                st.success("Model saved successfully!")
                st.code(f"Path: {save_path.resolve()}")
