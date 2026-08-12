"""
Pydantic schemas for API requests and responses.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from enum import Enum


# =============================================================================
# Auth Schemas
# =============================================================================
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: int
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# =============================================================================
# Preprocessing Schemas
# =============================================================================
class PreprocessingMethod(str, Enum):
    RAW = "raw"
    BUTTERWORTH = "butterworth"
    KALMAN = "kalman"
    BUTTERWORTH_KALMAN = "butterworth_kalman"
    VMD = "vmd"
    WAVELET = "wavelet"
    NOTCH = "notch"


class ButterworthConfig(BaseModel):
    order: int = 4
    low_hz: float = 10.0
    high_hz: float = 400.0


class KalmanConfig(BaseModel):
    q: float = 1e-5
    r: float = 0.05


class VMDConfig(BaseModel):
    alpha: int = 2000
    k: int = 3
    tau: int = 0
    dc: int = 0
    init: int = 1
    tol: float = 1e-7


class WaveletConfig(BaseModel):
    wavelet: str = "db4"
    level: int = 4
    threshold_mode: str = "soft"


class NotchConfig(BaseModel):
    freq: float = 50.0
    quality_factor: float = 30.0


class PreprocessingConfig(BaseModel):
    method: PreprocessingMethod = PreprocessingMethod.BUTTERWORTH_KALMAN
    butterworth: ButterworthConfig = ButterworthConfig()
    kalman: KalmanConfig = KalmanConfig()
    vmd: VMDConfig = VMDConfig()
    wavelet: WaveletConfig = WaveletConfig()
    notch: NotchConfig = NotchConfig()


class PreprocessRequest(BaseModel):
    config: PreprocessingConfig
    dataset_config: Optional[Dict[str, Any]] = None
    force_recompute: bool = False
    save_visualizations: bool = False


class PreprocessResponse(BaseModel):
    job_id: int
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: int
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    current_step: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class PreprocessingResult(BaseModel):
    job_id: int
    method: str
    config_hash: str
    channels: List[int]
    subjects: List[int]
    metrics: Dict[str, Any]
    data_path: Optional[str] = None
    scaler_path: Optional[str] = None
    visualization_paths: List[str] = []


# =============================================================================
# Model Schemas
# =============================================================================
class ModelCategory(str, Enum):
    CLASSICAL = "Classical ML"
    DEEP_LEARNING = "Deep Learning"
    TRANSFORMER = "Transformer"
    GRAPH = "Graph"
    EMBEDDED = "Embedded"


class ModelInfo(BaseModel):
    type: str
    name: str
    category: str
    is_classical: bool
    is_dl: bool
    supports_proba: bool
    default_hyperparameters: Dict[str, Any]


class HyperparameterSchema(BaseModel):
    name: str
    type: str  # int, float, string, bool, list
    default: Any
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[Any]] = None


class TrainRequest(BaseModel):
    model_type: str
    hyperparameters: Dict[str, Any] = {}
    preprocessing_job_id: Optional[int] = None
    preprocessing_config: Optional[PreprocessingConfig] = None
    training_config: Optional[Dict[str, Any]] = None
    save_model: bool = True
    skip_visualization: bool = False


class TrainResponse(BaseModel):
    job_id: int
    status: str
    message: str


class TrainingResult(BaseModel):
    job_id: int
    model_type: str
    prep_method: str
    metrics: Dict[str, Any]
    training_time: float
    inference_time: float
    epochs_run: int
    n_params: int
    trainable_params: int
    model_path: Optional[str] = None
    visualization_paths: List[str] = []
    mlflow_run_id: Optional[str] = None


# =============================================================================
# Experiment Schemas
# =============================================================================
class ExcludePair(BaseModel):
    prep: str
    model: str


class GridSearchRequest(BaseModel):
    preps: List[str] = []
    models: List[str] = []
    exclude_pairs: List[ExcludePair] = []
    skip_visualization: bool = False
    max_parallel: int = 2


class GridSearchResponse(BaseModel):
    experiment_id: int
    status: str
    message: str


class ExperimentResult(BaseModel):
    experiment_id: int
    name: str
    status: str
    total_combinations: int
    completed: int
    failed: int
    results: List[Dict[str, Any]]
    best_by_metric: Dict[str, Dict[str, Any]]
    total_time: float
    results_csv_path: Optional[str] = None


# =============================================================================
# Artifact Schemas
# =============================================================================
class ArtifactType(str, Enum):
    PREPROCESSED_DATA = "preprocessed_data"
    MODEL = "model"
    PLOT = "plot"
    REPORT = "report"
    TFLITE = "tflite"


class ArtifactResponse(BaseModel):
    id: int
    name: str
    type: str
    size_bytes: int
    mime_type: str
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    download_url: str

    class Config:
        from_attributes = True


class ArtifactListResponse(BaseModel):
    artifacts: List[ArtifactResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# WebSocket Schemas
# =============================================================================
class WSMessageType(str, Enum):
    JOB_UPDATE = "job_update"
    EMG_DATA = "emg_data"
    TRAINING_METRICS = "training_metrics"
    ERROR = "error"


class WSJobUpdate(BaseModel):
    type: str = "job_update"
    job_id: int
    job_type: str  # preprocessing, training, experiment
    status: str
    progress: float
    current_step: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class WSEMGData(BaseModel):
    type: str = "emg_data"
    session_id: str
    timestamp: float
    channels: List[float]  # Length = n_channels
    sample_rate: int


class WSTrainingMetrics(BaseModel):
    type: str = "training_metrics"
    job_id: int
    epoch: int
    metrics: Dict[str, float]
    epoch_time: float
    throughput: float


# =============================================================================
# System Schemas
# =============================================================================
class GPUInfo(BaseModel):
    name: str
    vendor: str
    vram_mb: Any


class SystemInfo(BaseModel):
    os: str
    cpu_name: str
    cpu_cores: int
    ram_gb: int
    gpus: List[GPUInfo]