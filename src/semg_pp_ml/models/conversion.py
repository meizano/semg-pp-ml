"""
Model conversion and Arduino export utilities for PyTorch models.
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Any
import json
import torch


def save_pytorch_model(model: Any, output_path: str | Path, model_name: str = "model") -> str:
    """Save PyTorch model in .pt format."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pt_path = output_path / f"{model_name}.pt"
    if hasattr(model, 'save'):
        model.save(pt_path)
    else:
        torch.save(model, pt_path)
    return str(pt_path)


def convert_to_onnx(
    model: Any,
    representative_data: np.ndarray,
    output_path: str | Path,
    model_name: str = "model"
) -> Tuple[str, str]:
    """
    Export PyTorch model to ONNX format for embedded/edge deployment.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_path = output_path / f"{model_name}.onnx"
    
    net = model.net if hasattr(model, 'net') else model
    net.eval()

    is_conv = any(c in str(type(net)).lower() for c in ['cnn', 'tcn', 'gcn', 'gat'])
    if is_conv and representative_data.ndim == 3:
        dummy_input = torch.tensor(representative_data[0:1], dtype=torch.float32).permute(0, 2, 1)
    else:
        dummy_input = torch.tensor(representative_data[0:1], dtype=torch.float32)

    device = next(net.parameters()).device
    dummy_input = dummy_input.to(device)

    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"[OK] ONNX Model: {onnx_path} ({size_kb:.2f} KB)")

    meta = {
        'model_name': model_name,
        'platform': 'Cross-Platform Edge / Microcontroller',
        'onnx_size_kb': round(size_kb, 2),
        'input_shape': list(dummy_input.shape),
    }
    meta_path = output_path / f"{model_name}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return str(onnx_path), str(meta_path)


def generate_arduino_sketch(
    config: dict,
    model_name: str = "tiny_nrf52840",
    output_path: Optional[str | Path] = None
) -> str:
    """Generate Arduino sketch."""
    from ..utils.arduino import generate_arduino_sketch as gen_sketch
    return gen_sketch(config, model_name, output_path)
