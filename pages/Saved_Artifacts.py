r"""
Saved Artifacts Explorer & Management
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

st.set_page_config(page_title="Saved Artifacts - sEMG Platform", page_icon="📦", layout="wide")

st.title("Saved Artifacts & Models")
st.caption("Manage preprocessed `.npz` dataset files, trained `.pt` / `.pkl` models, and experiment CSV logs.")

tab_data, tab_models, tab_grid = st.tabs([
    "Preprocessed Datasets (.npz)",
    "Trained Models (.pt / .pkl)",
    "Grid Search CSV Summaries"
])

# TAB 1: Preprocessed Datasets
with tab_data:
    st.subheader("Saved Preprocessed Datasets")
    prep_dir = Path("outputs/preprocessed_data")

    if not prep_dir.exists() or not list(prep_dir.glob("*.npz")):
        st.info("No saved `.npz` preprocessed datasets found in `outputs/preprocessed_data`.")
    else:
        npz_files = sorted(list(prep_dir.glob("*.npz")))

        for f in npz_files:
            with st.expander(f"{f.name} ({os.path.getsize(f) / (1024*1024):.2f} MB)", expanded=False):
                try:
                    data = np.load(f)
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**Shape**: `{data['emg_scaled'].shape}`")
                    col2.write(f"**Method**: `{data['method'] if 'method' in data else 'N/A'}`")
                    col3.write(f"**Channels**: `{data['channels'] if 'channels' in data else 'N/A'}`")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

                if st.button("Delete Dataset", key=f"del_npz_{f.name}"):
                    f.unlink(missing_ok=True)
                    f.with_name(f.stem + "_scaler.pkl").unlink(missing_ok=True)
                    f.with_name(f.stem + "_metrics.json").unlink(missing_ok=True)
                    st.success(f"Deleted {f.name}")
                    st.rerun()

# TAB 2: Trained Models
with tab_models:
    st.subheader("Trained Model Artifacts")
    model_dir = Path("outputs/trained_models")

    if not model_dir.exists() or not list(model_dir.glob("*.*")):
        st.info("No trained models found in `outputs/trained_models`.")
    else:
        model_files = sorted(list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pkl")))

        for mf in model_files:
            with st.expander(f"{mf.name} ({os.path.getsize(mf) / 1024:.1f} KB)", expanded=False):
                st.write(f"**File Path**: `{mf.resolve()}`")

                meta_json = mf.with_name(mf.stem + "_meta.json")
                if meta_json.exists():
                    with open(meta_json, "r") as jf:
                        meta = json.load(jf)
                    st.json(meta)

                if st.button("Delete Model", key=f"del_model_{mf.name}"):
                    mf.unlink(missing_ok=True)
                    meta_json.unlink(missing_ok=True)
                    st.success(f"Deleted {mf.name}")
                    st.rerun()

# TAB 3: Grid Search Logs
with tab_grid:
    st.subheader("Grid Search Summaries")
    grid_dir = Path("outputs/grid_results")

    if not grid_dir.exists() or not list(grid_dir.glob("*.csv")):
        st.info("No grid search summary CSV files found in `outputs/grid_results`.")
    else:
        csv_files = sorted(list(grid_dir.glob("*.csv")))
        for cf in csv_files:
            st.markdown(f"#### {cf.name}")
            df_grid = pd.read_csv(cf)
            st.dataframe(df_grid, use_container_width=True)
            csv_data = df_grid.to_csv(index=False).encode("utf-8")
            st.download_button(f"Download {cf.name}", data=csv_data, file_name=cf.name, mime="text/csv", key=f"dl_{cf.name}")
