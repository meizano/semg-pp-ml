import numpy as np
import pytest

from semg_pp_ml.augmentation import (
    SyntheticConfig,
    augment_dataset,
    add_gaussian_noise,
    electrode_shift,
    generate_synthetic_semg,
    mixup_same_class,
    time_warp,
)


@pytest.fixture
def windows():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 150, 8)).astype(np.float32)
    y = np.repeat(np.arange(4), 10)
    return X, y


def test_augment_shapes_and_labels(windows):
    X, y = windows
    Xa, ya = augment_dataset(X, y, n_copies=2, rng=1)
    assert Xa.shape == (X.shape[0] * 3, *X.shape[1:])
    assert ya.shape == (len(y) * 3,)
    assert np.array_equal(ya[: len(y)], y)
    assert np.isfinite(Xa).all()


def test_augment_is_reproducible(windows):
    X, y = windows
    a, _ = augment_dataset(X, y, n_copies=1, rng=42)
    b, _ = augment_dataset(X, y, n_copies=1, rng=42)
    assert np.array_equal(a, b)


def test_noise_hits_requested_snr(windows):
    X, _ = windows
    noisy = add_gaussian_noise(X, snr_db=10.0, rng=0)
    residual = noisy - X
    snr = 10 * np.log10(np.mean(X**2) / np.mean(residual**2))
    assert abs(snr - 10.0) < 0.5


def test_time_warp_keeps_length_and_is_identity_when_sigma_zero(windows):
    X, _ = windows
    assert time_warp(X, sigma=0.0, rng=0).shape == X.shape
    ident = time_warp(X, sigma=0.0, rng=0)
    assert np.allclose(ident, X, atol=1e-4)


def test_electrode_shift_zero_is_identity(windows):
    X, _ = windows
    assert np.array_equal(electrode_shift(X, max_shift=0, rng=0), X)


def test_mixup_keeps_labels_and_stays_in_class(windows):
    X, y = windows
    mixed, ym = mixup_same_class(X, y, rng=0)
    assert np.array_equal(ym, y)
    # a mixed window is a convex combination of same-class windows, so every value stays
    # inside that class's [min, max] range
    for c in np.unique(y):
        lo, hi = X[y == c].min(axis=0), X[y == c].max(axis=0)
        assert np.all(mixed[y == c] >= lo - 1e-5) and np.all(mixed[y == c] <= hi + 1e-5)


def test_unknown_op_raises(windows):
    X, y = windows
    with pytest.raises(ValueError):
        augment_dataset(X, y, ops=("does_not_exist",))


def test_synthetic_shapes_and_labels():
    cfg = SyntheticConfig(fs=1000.0, window_ms=100.0, n_channels=6, n_motor_units=20, seed=3)
    X, y, models = generate_synthetic_semg(n_classes=4, windows_per_class=5, cfg=cfg)
    assert X.shape == (20, 100, 6)
    assert X.dtype == np.float32
    assert np.bincount(y).tolist() == [5, 5, 5, 5]
    assert len(models) == 4
    assert np.isfinite(X).all()


def test_synthetic_is_reproducible():
    cfg = SyntheticConfig(fs=1000.0, window_ms=100.0, n_channels=4, n_motor_units=10, seed=7)
    a, ya, _ = generate_synthetic_semg(3, 4, cfg)
    b, yb, _ = generate_synthetic_semg(3, 4, cfg)
    assert np.array_equal(a, b) and np.array_equal(ya, yb)


def test_synthetic_rms_increases_with_activation_level():
    """Higher-force classes should produce larger EMG RMS (basic physiology sanity check)."""
    cfg = SyntheticConfig(fs=1000.0, window_ms=300.0, n_channels=8, n_motor_units=40, seed=11)
    X, y, models = generate_synthetic_semg(n_classes=5, windows_per_class=30, cfg=cfg)
    rms = np.array([np.sqrt(np.mean(X[y == c] ** 2)) for c in range(5)])
    levels = np.array([m.level for m in models])
    assert np.all(np.diff(levels) > 0)
    assert np.all(np.diff(rms) > 0), rms
