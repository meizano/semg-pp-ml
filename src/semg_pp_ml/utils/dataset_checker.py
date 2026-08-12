"""
Dataset checker and inspector for NinaPro DB8 (.mat files).
Explains classification (stimulus) vs regression (glove) paradigms.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import scipy.io as sio
from typing import Dict, Any, List, Optional


def inspect_ninapro_dataset(data_dir: str | Path, subjects: List[int] = [1]) -> Dict[str, Any]:
    """Inspect NinaPro DB8 dataset files and return summary metadata."""
    data_dir = Path(data_dir)
    report = {
        "data_dir": str(data_dir.resolve()),
        "exists": data_dir.exists(),
        "subjects_found": [],
        "total_files": 0,
        "sample_shape": None,
        "has_emg": False,
        "has_stimulus": False,
        "has_glove": False,
        "paradigm_guide": {
            "classification": "Uses discrete 'stimulus' / 'restimulus' labels (e.g., gesture classes 0 to N). Loss: CrossEntropyLoss.",
            "regression": "Uses continuous 'glove' (CyberGlove joint angles) for DOF kinematics decoding. Loss: MSE / MAE."
        }
    }

    if not data_dir.exists():
        return report

    for subj in subjects:
        pattern = data_dir / f"S{subj}_*.mat"
        files = list(data_dir.glob(f"S{subj}_*.mat"))
        if files:
            report["subjects_found"].append(subj)
            report["total_files"] += len(files)
            # Inspect first file
            try:
                mat = sio.loadmat(files[0])
                if 'emg' in mat or 'EMG' in mat:
                    report["has_emg"] = True
                    emg = mat.get('emg', mat.get('EMG'))
                    report["sample_shape"] = emg.shape
                if 'stimulus' in mat or 'restimulus' in mat:
                    report["has_stimulus"] = True
                if 'glove' in mat:
                    report["has_glove"] = True
            except Exception:
                pass

    return report
