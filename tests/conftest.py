"""Shared fixtures: a small NinaPro-style dataset written to disk as .mat files."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from semg_pp_ml.augmentation import SyntheticConfig, generate_synthetic_semg


def write_fake_ninapro(folder: Path, subjects=(1, 2), per_class: int = 30, seed: int = 0) -> Path:
    """Write S<n>_A1_E<k>.mat files shaped like NinaPro DB8 (emg, stimulus, glove).

    Labels: 0 = rest, 1..6 = gestures. Windows from the synthetic generator are
    concatenated into continuous segments so the sliding-window code sees realistic input.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for subj in subjects:
        cfg = SyntheticConfig(fs=2000, window_ms=200, n_channels=14, n_motor_units=20, seed=subj)
        X, y, _ = generate_synthetic_semg(6, per_class, cfg)   # (n, 400, 14), labels 0..5
        for rep in (1, 2):
            order = rng.permutation(len(y))[: len(y) // 2]
            emg = np.concatenate([X[i] for i in order]).astype(np.float64)
            stim = np.concatenate([np.full(400, y[i] + 1) for i in order]).reshape(-1, 1)
            glove = rng.standard_normal((len(stim), 22))
            sio.savemat(folder / f"S{subj}_A1_E{rep}.mat", {"emg": emg, "stimulus": stim, "glove": glove})
    return folder


@pytest.fixture(scope="session")
def fake_dataset_root(tmp_path_factory) -> Path:
    """Dataset root with the files in ``train/`` (matches the default layout)."""
    root = tmp_path_factory.mktemp("ninapro")
    write_fake_ninapro(root / "train")
    return root
