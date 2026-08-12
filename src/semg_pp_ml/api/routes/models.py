"""
Model training routes.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_current_user, get_config
from ..schemas import (
    TrainRequest, TrainResponse, JobStatus, TrainingResult,
    ModelInfo, HyperparameterSchema, ModelCategory
)
from ..models.db import TrainingJob, PreprocessingJob, Artifact, User
from ...models.registry import (
    get_available_models, get_model_categories, get_model_names,
    get_model_info, get_default_hyperparameters, merge_hyperparameters,
    is_classical_model, build_model, count_params, save_model
)
from ...experiments.runner import run_single_experiment, ExperimentResult
from ...config import ExperimentConfig

router = APIRouter()

job_status = {}


def update_job_status(job_id: int, status: str, progress: float = 0, step: str = None, metrics: dict = None, error: str = None):
    """Update job status and notify WebSocket."""
    from ...api.services.websocket_manager import manager
    import asyncio
    
    job_status[job_id] = {
        "status": status,
        "progress": progress,
        "current_step": step,
        "metrics": metrics,
        "error": error
    }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(manager.send_job_update(str(job_id), status, progress, metrics))
    except:
        pass


async def run_training_job(
    job_id: int,
    config: ExperimentConfig,
    model_type: str,
    hyperparameters: Dict[str, Any],
    preprocessing_job_id: Optional[int],
    user_id: int,
    db: Session,
    save_model_flag: bool,
    skip_visualization: bool
):
    """Background task to run model training."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        return

    try:
        update_job_status(job_id, "running", 0.1, "Preparing data")
        
        # Get preprocessing data if specified
        prep_data = None
        if preprocessing_job_id:
            prep_job = db.query(PreprocessingJob).filter(
                PreprocessingJob.id == preprocessing_job_id,
                PreprocessingJob.user_id == user_id
            ).first()
            if prep_job and prep_job.artifact:
                # Load preprocessed data
                import numpy as np
                import joblib
                data = np.load(prep_job.artifact.storage_path)
                scaler = joblib.load(prep_job.artifact.storage_path.replace(".npz", "_scaler.pkl"))
                prep_data = {
                    'emg_scaled': data['emg_scaled'],
                    'emg_processed': data['emg_processed'],
                    'scaler': scaler,
                    'config_hash': data['config_hash'].item() if isinstance(data['config_hash'], np.ndarray) else data['config_hash'],
                    'method': data['method'].item() if isinstance(data['method'], np.ndarray) else data['method'],
                    'channels': data['channels'].tolist(),
                    'subjects': data['subjects'].tolist(),
                }
        
        update_job_status(job_id, "running", 0.3, "Building model")
        
        # Prepare config for training
        train_config = config.model_copy(deep=True)
        train_config.model = type('obj', (object,), merge_hyperparameters(model_type, config.model.model_dump()))()
        
        if prep_data:
            # Use preprocessed data - need to modify run_single_experiment
            # For now, run full pipeline with the preprocessing method
            train_config.preprocessing.method = prep_data['method']
        
        update_job_status(job_id, "running", 0.5, "Training model")
        
        # Run experiment
        result = run_single_experiment(
            train_config,
            prep_method=train_config.preprocessing.method,
            model_type=model_type,
            save_model_flag=save_model_flag,
            skip_visualization=skip_visualization,
            output_dir=Path(config.output_dir)
        )
        
        update_job_status(job_id, "running", 0.9, "Saving results")
        
        # Save model if requested
        model_artifact = None
        if save_model_flag and result.model_path:
            model_artifact = Artifact(
                name=Path(result.model_path).name,
                type="model",
                storage_path=result.model_path,
                size_bytes=os.path.getsize(result.model_path),
                mime_type="application/octet-stream",
                metadata={
                    "model_type": model_type,
                    "n_params": result.n_params,
                    "input_shape": list(train_config.signal.window_size, train_config.dataset.n_channels),
                }
            )
            db.add(model_artifact)
            db.flush()
        
        # Update job
        job.status = "completed"
        job.metrics = result.metrics
        job.model_artifact_id = model_artifact.id if model_artifact else None
        job.completed_at = datetime.utcnow()
        db.commit()
        
        update_job_status(job_id, "completed", 1.0, "Done", result.metrics)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
        update_job_status(job_id, "failed", 0, error=str(e))


@router.get("/", response_model=List[ModelInfo])
def list_models(current_user: User = Depends(get_current_user)):
    """List all available models with metadata."""
    categories = get_model_categories()
    names = get_model_names()
    
    models = []
    for model_type in get_available_models():
        info = get_model_info(model_type)
        defaults = get_default_hyperparameters(model_type)
        models.append(ModelInfo(
            type=model_type,
            name=names.get(model_type, model_type),
            category=info['category'],
            is_classical=info['is_classical'],
            is_dl=info['is_dl'],
            supports_proba=info['supports_proba'],
            default_hyperparameters=defaults
        ))
    return models


@router.get("/categories")
def list_model_categories(current_user: User = Depends(get_current_user)):
    """Get model categories for UI grouping."""
    return get_model_categories()


@router.get("/{model_type}/hyperparameters", response_model=List[HyperparameterSchema])
def get_model_hyperparameters(
    model_type: str,
    current_user: User = Depends(get_current_user)
):
    """Get hyperparameter schema for a model."""
    info = get_model_info(model_type)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    defaults = get_default_hyperparameters(model_type)
    
    schemas = []
    for param, default in defaults.items():
        schemas.append(HyperparameterSchema(
            name=param,
            type=type(default).__name__,
            default=default,
            description=f"Hyperparameter: {param}"
        ))
    return schemas


@router.post("/train", response_model=TrainResponse)
def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    config: ExperimentConfig = Depends(get_config)
):
    """Start a model training job."""
    # Validate model type
    available = get_available_models()
    if request.model_type not in available:
        raise HTTPException(status_code=400, detail=f"Unknown model type. Available: {available}")
    
    # Validate preprocessing job if specified
    if request.preprocessing_job_id:
        prep_job = db.query(PreprocessingJob).filter(
            PreprocessingJob.id == request.preprocessing_job_id,
            PreprocessingJob.user_id == current_user.id
        ).first()
        if not prep_job:
            raise HTTPException(status_code=404, detail="Preprocessing job not found")
        if prep_job.status != "completed":
            raise HTTPException(status_code=400, detail="Preprocessing job not completed")
    
    # Create job record
    job = TrainingJob(
        user_id=current_user.id,
        name=f"{request.model_type}_{uuid.uuid4().hex[:8]}",
        model_type=request.model_type,
        hyperparameters=request.hyperparameters,
        preprocessing_job_id=request.preprocessing_job_id,
        config=request.model_dump(),
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    job_status[job.id] = {"status": "pending", "progress": 0}
    
    background_tasks.add_task(
        run_training_job,
        job.id, config, request.model_type, request.hyperparameters,
        request.preprocessing_job_id, current_user.id, db,
        request.save_model, request.skip_visualization
    )
    
    return TrainResponse(
        job_id=job.id,
        status="pending",
        message="Training job started"
    )


@router.get("/jobs", response_model=List[JobStatus])
def list_training_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all training jobs for current user."""
    jobs = db.query(TrainingJob).filter(TrainingJob.user_id == current_user.id).order_by(TrainingJob.created_at.desc()).all()
    
    result = []
    for job in jobs:
        status_info = job_status.get(job.id, {"status": job.status, "progress": 1.0 if job.status == "completed" else 0})
        result.append(JobStatus(
            job_id=job.id,
            status=job.status,
            progress=status_info.get("progress", 0),
            current_step=status_info.get("current_step"),
            metrics=status_info.get("metrics"),
            error=status_info.get("error"),
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        ))
    return result


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_training_job_status(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get status of a specific training job."""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_info = job_status.get(job_id, {"status": job.status, "progress": 1.0 if job.status == "completed" else 0})
    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=status_info.get("progress", 0),
        current_step=status_info.get("current_step"),
        metrics=status_info.get("metrics"),
        error=status_info.get("error"),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.get("/jobs/{job_id}/result", response_model=TrainingResult)
def get_training_result(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get training job result."""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    return TrainingResult(
        job_id=job.id,
        model_type=job.model_type,
        prep_method=job.config.get("config", {}).get("preprocessing", {}).get("method", "unknown"),
        metrics=job.metrics or {},
        training_time=0,  # Could be stored in job
        inference_time=0,
        epochs_run=job.metrics.get("epochs_run", 0) if job.metrics else 0,
        n_params=job.metrics.get("model_params_total", 0) if job.metrics else 0,
        trainable_params=job.metrics.get("model_params_trainable", 0) if job.metrics else 0,
        model_path=job.model_artifact.storage_path if job.model_artifact else None,
        visualization_paths=[],
        mlflow_run_id=job.mlflow_run_id
    )