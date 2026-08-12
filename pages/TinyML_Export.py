r"""
TinyML Export - ONNX Export & Arduino Sketch Generation
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import numpy as np
import json

from semg_pp_ml.models.conversion import convert_to_onnx, generate_arduino_sketch
from semg_pp_ml.models.registry import load_model

st.set_page_config(page_title="TinyML Export - sEMG Platform", page_icon="⚡", layout="wide")

st.title("TinyML & Embedded Export")
st.caption("Convert trained PyTorch models to ONNX format and auto-generate Arduino C++ code (`.ino` / `.h`) for embedded deployment.")

st.subheader("1. Select Trained Model File")

models_dir = Path("outputs/trained_models")
model_files = sorted(list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pkl"))) if models_dir.exists() else []

if not model_files:
    st.warning("No trained model files (`.pt` or `.pkl`) found in `outputs/trained_models`. Please train a model first.")
    st.stop()

selected_model_path = st.selectbox("Choose Model File", model_files, format_func=lambda p: p.name)

st.subheader("2. Target Microcontroller Settings")
col1, col2 = st.columns(2)
with col1:
    model_name_base = st.text_input("Export Base Name", selected_model_path.stem)
with col2:
    n_channels = st.number_input("Number of sEMG Channels", value=6, min_value=1, max_value=16)
    window_size = st.number_input("Window Size (timesteps)", value=400, min_value=10, max_value=2000)
    n_classes = st.number_input("Number of Gesture Classes", value=6, min_value=2, max_value=20)

st.subheader("3. Arduino Pin Configuration")
pin_input = st.text_input("ADC Pins List (comma separated)", "A0, A1, A2, A3, A4, A5")
emg_pins = [p.strip() for p in pin_input.split(",") if p.strip()]

out_export_dir = Path("outputs/tinyml_export")

if st.button("Convert Model to ONNX & Generate Arduino Sketch", type="primary", use_container_width=True):
    with st.spinner("Loading model and exporting to ONNX..."):
        try:
            model_type = selected_model_path.stem.split('_')[1] if len(selected_model_path.stem.split('_')) > 1 else 'mlp'
            
            if selected_model_path.suffix == '.pkl':
                import joblib
                model = joblib.load(selected_model_path)
            else:
                model = load_model(selected_model_path, model_type)

            input_shape = (100, window_size, n_channels)
            rep_data = np.random.normal(0, 1, input_shape).astype(np.float32)

            out_export_dir.mkdir(parents=True, exist_ok=True)
            onnx_path, meta_path = convert_to_onnx(model=model, representative_data=rep_data, output_path=out_export_dir, model_name=model_name_base)

            sketch_cfg = {"n_channels": n_channels, "window_size": window_size, "n_classes": n_classes, "emg_pins": emg_pins}
            sketch_path = out_export_dir / f"{model_name_base}_arduino.ino"
            sketch_content = generate_arduino_sketch(config=sketch_cfg, model_name=model_name_base, output_path=sketch_path)

            st.success("Conversion & Code Generation Successful!")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("ONNX Model Size", f"{os.path.getsize(onnx_path) / 1024:.2f} KB")
                st.code(f"ONNX File : {Path(onnx_path).resolve()}\nMetadata  : {Path(meta_path).resolve()}")
            with c2:
                st.metric("Arduino Sketch Size", f"{os.path.getsize(sketch_path) / 1024:.2f} KB")
                st.code(f"Arduino Sketch : {Path(sketch_path).resolve()}")

            st.markdown("#### Generated Arduino Sketch (.ino)")
            st.code(sketch_content, language="cpp")

        except Exception as e:
            st.error(f"Conversion failed: {e}")
            import traceback
            st.code(traceback.format_exc())
