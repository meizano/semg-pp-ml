"""
Configuration models for sEMG Preprocessing and ML Platform.
Uses Pydantic v2 for validation and YAML for file-based config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Dict, Any, List, get_origin
from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml


class _Base(BaseModel):
    """Shared settings: validate on assignment and reject unknown keys (catches YAML typos)."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True, extra="forbid")


# =============================================================================
# Dataset Configuration
# =============================================================================
DEFAULT_DATA_ROOT = Path("data/ninapro_db8")


class DatasetConfig(_Base):
    root: Path = DEFAULT_DATA_ROOT
    train_dir: str = "train"        # relative to root, or an absolute path
    eval_dir: str = "evaluation"    # optional; used only if it contains .mat files
    subjects: List[int] = Field(default_factory=lambda: list(range(1, 11)))
    channels: List[int] = [3, 4, 5, 11, 12, 13]
    n_channels: int = 6
    classes: List[int] = [0, 1, 2, 3, 4, 5]
    n_classes: int = 6

    @field_validator("root", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    def resolve_dir(self, subdir: str | Path) -> Path:
        """Absolute paths are used as-is; relative paths are taken under ``root``."""
        p = Path(subdir).expanduser()
        return p if p.is_absolute() else (self.root / p)

    @property
    def train_path(self) -> Path:
        """``root/train_dir``. If that subfolder is missing but ``root`` itself holds
        NinaPro ``S*.mat`` files, ``root`` is used, so pointing ``--data-dir`` at the
        folder with the files just works."""
        path = self.resolve_dir(self.train_dir)
        if not path.is_dir() and self.root.is_dir() and any(self.root.glob("S*.mat")):
            return self.root
        return path

    @property
    def eval_path(self) -> Path:
        return self.resolve_dir(self.eval_dir)


# =============================================================================
# Signal Configuration
# =============================================================================
class SignalConfig(_Base):
    fs: int = 2000
    window_size: int = 400
    overlap: int = 100

    @property
    def step(self) -> int:
        return self.window_size - self.overlap


# =============================================================================
# Preprocessing Method Configurations
# =============================================================================
class ButterworthConfig(_Base):
    order: int = 4
    low_hz: float = 10.0
    high_hz: float = 400.0


class KalmanConfig(_Base):
    q: float = 1e-5
    r: float = 0.05


class VMDConfig(_Base):
    alpha: int = 2000
    k: int = 3
    tau: int = 0
    dc: int = 0
    init: int = 1
    tol: float = 1e-7


class WaveletConfig(_Base):
    wavelet: str = "db4"
    level: int = 4
    threshold_mode: Literal["soft", "hard"] = "soft"


class NotchConfig(_Base):
    freq: float = 50.0
    quality_factor: float = 30.0


PreprocessingMethod = Literal[
    "raw",
    "butterworth",
    "kalman",
    "butterworth_kalman",
    "vmd",
    "wavelet",
    "notch",
]


class PreprocessingConfig(_Base):
    method: PreprocessingMethod = "butterworth_kalman"
    butterworth: ButterworthConfig = Field(default_factory=ButterworthConfig)
    kalman: KalmanConfig = Field(default_factory=KalmanConfig)
    vmd: VMDConfig = Field(default_factory=VMDConfig)
    wavelet: WaveletConfig = Field(default_factory=WaveletConfig)
    notch: NotchConfig = Field(default_factory=NotchConfig)


# =============================================================================
# Training Configuration
# =============================================================================
class TrainingConfig(_Base):
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.001
    early_stopping: bool = True
    patience: int = 15
    test_size: float = 0.2
    random_state: int = 42
    validation_split: float = 0.2


# =============================================================================
# Model Hyperparameters (per-model)
# =============================================================================
class ModelHyperparameters(_Base):
    type: str = "cnn1d"
    # Classical ML
    svm: Dict[str, Any] = Field(default_factory=lambda: {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True})
    rf: Dict[str, Any] = Field(default_factory=lambda: {"n_estimators": 100, "max_depth": None, "random_state": 42, "n_jobs": -1})
    knn: Dict[str, Any] = Field(default_factory=lambda: {"n_neighbors": 5, "weights": "uniform", "n_jobs": -1})
    lda: Dict[str, Any] = Field(default_factory=lambda: {"solver": "svd"})
    nb: Dict[str, Any] = Field(default_factory=dict)
    mlp_sklearn: Dict[str, Any] = Field(default_factory=lambda: {"hidden_layer_sizes": (256, 128), "max_iter": 500, "random_state": 42})

    # Deep Learning - Shared
    activation: str = "relu"
    dropout_rate: float = 0.3
    dense_units: int = 128

    # CNN
    cnn_filters_1: int = 64
    cnn_filters_2: int = 128
    cnn_filters_3: int = 256
    cnn_kernel: int = 3
    pool_size: int = 2

    # RNN/LSTM/GRU
    rnn_units_1: int = 128
    rnn_units_2: int = 64

    # TCN
    tcn_filters: int = 64
    tcn_kernel_size: int = 3
    tcn_dilations: List[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])

    # Transformer
    transformer_heads: int = 4
    transformer_ff_dim: int = 128
    attention_key_dim: int = 64

    # MLP (Keras)
    mlp_units_1: int = 256
    mlp_units_2: int = 128

    # Embedded/Tiny models
    tiny_mlp_units: List[int] = Field(default_factory=lambda: [48, 24])
    tiny_cnn_filters: List[int] = Field(default_factory=lambda: [16, 8])
    tiny_cnn_kernels: List[int] = Field(default_factory=lambda: [5, 3])


# =============================================================================
# Experiment Configuration
# =============================================================================
class ExperimentConfig(_Base):
    name: str = "default"
    description: str = ""
    tags: Dict[str, str] = Field(default_factory=dict)

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelHyperparameters = Field(default_factory=ModelHyperparameters)

    output_dir: Path = Path("outputs")
    experiment_name: str = "semg_experiment"

    @field_validator("output_dir", mode="before")
    @classmethod
    def expand_output_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


# =============================================================================
# Grid Search Configuration
# =============================================================================
class GridSearchConfig(_Base):
    preps: List[PreprocessingMethod] = Field(default_factory=lambda: [
        "raw", "butterworth", "kalman", "butterworth_kalman", "vmd"
    ])
    models: List[str] = Field(default_factory=lambda: [
        "svm", "rf", "knn", "lda", "nb",
        "mlp", "cnn1d", "cnn2d", "lstm", "bilstm",
        "gru_lstm", "cnn_lstm", "tcn",
        "transformer", "cnn_transformer", "mafn",
        "msemg", "gcn", "gat"
    ])
    exclude_pairs: List[List[str]] = Field(default_factory=lambda: [
        ["raw", "lda"], ["raw", "svm"], ["raw", "knn"], ["raw", "nb"]
    ])
    skip_visualization: bool = False
    max_parallel: int = 2


# =============================================================================
# Config Loader
# =============================================================================
DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    return data


def load_config(config_path: str | Path | None = None) -> ExperimentConfig:
    """Load an experiment config from YAML.

    With no path, ``config/default.yaml`` (relative to the working directory) is used
    if it exists; otherwise the built-in defaults are returned.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        if config_path:
            raise FileNotFoundError(f"config file not found: {path}")
        return ExperimentConfig()
    return ExperimentConfig(**_read_yaml(path))


def load_grid_config(config_path: str | Path | None = None) -> GridSearchConfig:
    """Load a grid-search config (preps/models/exclude_pairs). Defaults if no path is given."""
    if not config_path:
        return GridSearchConfig()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"grid config file not found: {path}")
    return GridSearchConfig(**_read_yaml(path))


def apply_overrides(config: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    """Return a copy of ``config`` with ``dotted.key=value`` overrides applied.

    Values are parsed as YAML, so ``1e-3``, ``[1, 2]``, ``true`` and ``null`` work.
    Unknown keys raise ``KeyError`` instead of being silently ignored.

        apply_overrides(cfg, ["training.epochs=5", "preprocessing.method=vmd"])
    """
    data = config.model_dump(mode="python")
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        parts = key.strip().split(".")
        node = data
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"unknown config section {part!r} in override {key!r}")
            node = node[part]
        if parts[-1] not in node and not _inside_free_dict(parts[:-1]):
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return ExperimentConfig(**data)


def _inside_free_dict(parent: List[str]) -> bool:
    """True when the path ``parent`` ends inside a free-form Dict field (e.g. ``model.svm``, ``tags``)."""
    cls: type = ExperimentConfig
    for part in parent:
        field = cls.model_fields.get(part)
        if field is None:
            return False
        ann = field.annotation
        if get_origin(ann) is dict:
            return True
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            cls = ann
            continue
        return False
    return False


def save_config(config: ExperimentConfig, config_path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)


def create_default_configs(config_dir: str | Path = "config") -> None:
    """Create default configuration files."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Main config
    main_config = ExperimentConfig()
    save_config(main_config, config_dir / "default.yaml")

    # Per-preprocessing configs
    prep_configs = {
        "butterworth.yaml": {"preprocessing": {"method": "butterworth"}},
        "kalman.yaml": {"preprocessing": {"method": "kalman"}},
        "butterworth_kalman.yaml": {"preprocessing": {"method": "butterworth_kalman"}},
        "vmd.yaml": {"preprocessing": {"method": "vmd"}},
        "wavelet.yaml": {"preprocessing": {"method": "wavelet"}},
        "notch.yaml": {"preprocessing": {"method": "notch"}},
    }

    for filename, overrides in prep_configs.items():
        config = ExperimentConfig(**overrides)
        save_config(config, config_dir / "preprocessing" / filename)

    # Per-model configs
    model_configs = {
        "lstm.yaml": {"model": {"rnn_units_1": 128, "rnn_units_2": 64}},
        "bilstm.yaml": {"model": {"rnn_units_1": 128, "rnn_units_2": 64}},
        "cnn1d.yaml": {"model": {"cnn_filters_1": 64, "cnn_filters_2": 128, "cnn_filters_3": 256}},
        "tcn.yaml": {"model": {"tcn_filters": 64, "tcn_dilations": [1, 2, 4, 8, 16]}},
        "transformer.yaml": {"model": {"transformer_heads": 4, "transformer_ff_dim": 128}},
        "cnn_lstm.yaml": {"model": {"cnn_filters_1": 64, "cnn_filters_2": 128, "rnn_units_2": 64}},
        "svm.yaml": {"model": {"svm": {"C": 1.0, "kernel": "rbf"}}},
        "rf.yaml": {"model": {"rf": {"n_estimators": 100}}},
    }

    for filename, overrides in model_configs.items():
        config = ExperimentConfig(**overrides)
        save_config(config, config_dir / "models" / filename)

    # Grid search config
    grid_config = GridSearchConfig()
    save_config(grid_config, config_dir / "grid_search.yaml")


if __name__ == "__main__":
    create_default_configs()
    print("Default configurations created in config/")