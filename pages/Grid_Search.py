r"""
Grid Search & Batch Experimentation Dashboard
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from semg_pp_ml.config import ExperimentConfig, GridSearchConfig
from semg_pp_ml.experiments.grid_search import run_grid_search_optimized
from semg_pp_ml.models.registry import MODEL_NAMES

st.set_page_config(page_title="Grid Search - sEMG Platform", page_icon="🔍", layout="wide")

st.title("Grid Search & Automated Experimentation")
st.caption("Perform automated matrix experiments (`Preprocessing Methods × Models`). Data filtering is executed ONCE per method to maximize efficiency.")

# -----------------------------------------------------------------------------
# Configuration Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Experiment Matrix")

selected_preps = st.sidebar.multiselect(
    "Select Preprocessing Methods",
    ["raw", "butterworth", "kalman", "butterworth_kalman", "vmd", "wavelet", "notch"],
    default=["butterworth", "kalman", "butterworth_kalman"]
)

selected_models = st.sidebar.multiselect(
    "Select Models",
    ["svm", "rf", "knn", "lda", "nb", "mlp", "cnn1d", "cnn2d", "lstm", "bilstm", "gru_lstm", "cnn_lstm", "tcn", "transformer", "cnn_transformer", "mafn", "msemg", "gcn", "gat"],
    default=["svm", "rf", "cnn1d", "tcn"]
)

st.sidebar.header("Exclusion Rules")
exclude_raw = st.sidebar.checkbox("Skip raw data with classical models (LDA/SVM/KNN/NB)", value=True)

st.sidebar.header("Shared Training Config")
data_dir = st.sidebar.text_input("Train Data Folder", r"data\ninapro_db8")
subjects = st.sidebar.multiselect("Subjects", list(range(1, 11)), default=[1])
channels = st.sidebar.multiselect("Channels (0-indexed)", list(range(16)), default=[3, 4, 5, 11, 12, 13])
epochs = st.sidebar.number_input("Epochs", value=30, min_value=1, max_value=200)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)

out_grid_dir = Path("outputs/grid_results")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if not selected_preps:
    st.warning("Select at least one preprocessing method.")
    st.stop()

if not selected_models:
    st.warning("Select at least one model.")
    st.stop()

exclude_pairs = []
if exclude_raw:
    exclude_pairs = [("raw", "lda"), ("raw", "svm"), ("raw", "knn"), ("raw", "nb")]

total_combos = len(selected_preps) * len(selected_models) - len([p for p in exclude_pairs if p[0] in selected_preps and p[1] in selected_models])
st.info(f"Total Experiment Combinations to Run: **{total_combos}**")

run_btn = st.button("Start Grid Search Experiment", type="primary", use_container_width=True)

if run_btn or "grid_results_cache" in st.session_state:
    if run_btn:
        cfg = ExperimentConfig()
        cfg.dataset.train_dir = data_dir
        cfg.dataset.subjects = subjects
        cfg.dataset.channels = channels
        cfg.training.epochs = epochs
        cfg.training.batch_size = batch_size
        cfg.output_dir = out_grid_dir

        grid_cfg = GridSearchConfig(
            preps=selected_preps,
            models=selected_models,
            exclude_pairs=[list(p) for p in exclude_pairs]
        )

        with st.spinner("Running Grid Search... Data preprocessing runs 1x per filter method!"):
            grid_res = run_grid_search_optimized(cfg, grid_cfg, skip_visualization=True)
            st.session_state["grid_results_cache"] = grid_res.summary_df

    df = st.session_state["grid_results_cache"]

    st.success(f"Grid Search finished! Completed **{len(df)}** runs.")

    # Display Results Table
    st.subheader("Results Summary Table")

    disp_cols = [c for c in ["prep", "model", "accuracy", "f1_score", "precision", "recall", "specificity_mean", "inference_latency_ms", "model_params_total", "status"] if c in df.columns]

    subset_cols = [c for c in ["accuracy", "f1_score"] if c in df.columns]
    if subset_cols and not df.empty:
        st.dataframe(df[disp_cols].style.highlight_max(subset=subset_cols, color="#dcfce7"), use_container_width=True)
    else:
        st.dataframe(df[disp_cols], use_container_width=True)

    # Visualization
    if "accuracy" in df.columns and not df.empty:
        st.subheader("Comparative Performance")

        col1, col2 = st.columns(2)

        with col1:
            fig_acc = px.bar(
                df,
                x="model",
                y="accuracy",
                color="prep",
                barmode="group",
                title="Accuracy Comparison by Preprocessing & Model",
                labels={"model": "Model", "accuracy": "Accuracy", "prep": "Preprocessing"}
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            if "f1_score" in df.columns:
                fig_f1 = px.bar(
                    df,
                    x="model",
                    y="f1_score",
                    color="prep",
                    barmode="group",
                    title="F1-Score Comparison",
                    labels={"model": "Model", "f1_score": "F1 Score", "prep": "Preprocessing"}
                )
                st.plotly_chart(fig_f1, use_container_width=True)

    # Download CSV
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export Grid Search Results (CSV)", data=csv_data, file_name="grid_search_summary.csv", mime="text/csv")
