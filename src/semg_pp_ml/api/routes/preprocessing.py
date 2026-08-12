"""
Preprocessing routes for sEMG signal processing.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_current_user, get_config
from ..schemas import (
    PreprocessRequest, PreprocessResponse, JobStatus, PreprocessingResult,
    PreprocessingMethod, PreprocessingConfig
)
from ..models.db import PreprocessingJob, Artifact, User
from ...preprocessing.pipeline import PreprocessingPipeline
from ...config import ExperimentConfig

router = APIRouter()

# In-memory job tracking (replace with Redis in production)
job_status = {}


def update_job_status(job_id: int, status: str, progress: float = 0, step: str = None, metrics: dict = None, error: str = None):
    """Update job status in memory and notify WebSocket."""
    from ...api.services.websocket_manager import manager
    import asyncio
    
    job_status[job_id] = {
        "status": status,
        "progress": progress,
        "current_step": step,
        "metrics": metrics,
        "error": error
    }
    
    # Notify WebSocket
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(manager.send_job_update(str(job_id), status, progress, metrics))
    except:
        pass


async def run_preprocessing_job(
    job_id: int,
    config: ExperimentConfig,
    user_id: int,
    db: Session
):
    """Background task to run preprocessing."""
    job = db.query(PreprocessingJob).filter(PreprocessingJob.id == job_id).first()
    if not job:
        return

    try:
        update_job_status(job_id, "running", 0.1, "Initializing pipeline")
        
        # Run preprocessing
        pipeline = PreprocessingPipeline(config)
        update_job_status(job_id, "running", 0.3, "Loading and filtering data")
        
        result = pipeline.run(
            method=config.preprocessing.method,
            force_recompute=job.config.get("force_recompute", False),
            save_visualizations=job.config.get("save_visualizations", False)
        )
        
        update_job_status(job_id, "running", 0.8, "Saving results")
        
        # Save artifacts
        output_dir = Path(config.output_dir) / "preprocessing"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import numpy as np
        import joblib
        import json
        
        data_filename = f"preprocessed_{result.config_hash}.npz"
        data_path = output_dir / data_filename
        np.savez_compressed(
            data_path,
            emg_processed=result.emg_processed,
            emg_scaled=result.emg_scaled,
            config_hash=result.config_hash,
            method=result.method,
            channels=np.array(result.channels),
            subjects=np.array(result.subjects),
        )
        
        scaler_filename = f"scaler_{result.config_hash}.pkl"
        scaler_path = output_dir / scaler_filename
        joblib.dump(result.scaler, scaler_path)
        
        metrics_filename = f"metrics_{result.config_hash}.json"
        metrics_path = output_dir / metrics_filename
        with open(metrics_path, 'w') as f:
            json.dump(result.metrics, f, indent=2)
        
        # Create artifact records
        data_artifact = Artifact(
            name=data_filename,
            type="preprocessed_data",
            storage_path=str(data_path),
            size_bytes=os.path.getsize(data_path),
            mime_type="application/octet-stream",
            metadata={
                "config_hash": result.config_hash,
                "method": result.method,
                "channels": result.channels,
                "subjects": result.subjects,
            }
        )
        db.add(data_artifact)
        db.flush()
        
        scaler_artifact = Artifact(
            name=scaler_filename,
            type="preprocessed_data",
            storage_path=str(scaler_path),
            size_bytes=os.path.getsize(scaler_path),
            mime_type="application/octet-stream",
        )
        db.add(scaler_artifact)
        
        metrics_artifact = Artifact(
            name=metrics_filename,
            type="preprocessed_data",
            storage_path=str(metrics_path),
            size_bytes=os.path.getsize(metrics_path),
            mime_type="application/json",
        )
        db.add(metrics_artifact)
        
        # Update job
        job.status = "completed"
        job.output_artifact_id = data_artifact.id
        job.metrics = result.metrics
        job.completed_at = datetime.utcnow()
        db.commit()
        
        update_job_status(job_id, "completed", 1.0, "Done", result.metrics)
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
        update_job_status(job_id, "failed", 0, error=str(e))


@router.get("/methods", response_model=List[dict])
def list_preprocessing_methods(current_user: User = Depends(get_current_user)):
    """List available preprocessing methods with their parameters."""
    methods = []
    for method in PreprocessingMethod:
        methods.append({
            "value": method.value,
            "name": method.value.replace("_", " ").title(),
            "description": f"{method.value} preprocessing"
        })
    return methods


@router.post("/run", response_model=PreprocessResponse)
def run_preprocessing(
    request: PreprocessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    config: ExperimentConfig = Depends(get_config)
):
    """Start a preprocessing job."""
    # Merge configs
    merged_config = config.model_copy(deep=True)
    merged_config.preprocessing = request.config
    
    if request.dataset_config:
        # Override dataset config if provided
        pass
    
    # Create job record
    job = PreprocessingJob(
        user_id=current_user.id,
        name=f"{request.config.method}_{uuid.uuid4().hex[:8]}",
        config=request.model_dump(),
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Initialize status
    job_status[job.id] = {"status": "pending", "progress": 0}
    
    # Run in background
    background_tasks.add_task(run_preprocessing_job, job.id, merged_config, current_user.id, db)
    
    return PreprocessResponse(
        job_id=job.id,
        status="pending",
        message="Preprocessing job started"
    )


@router.get("/jobs", response_model=List[JobStatus])
def list_preprocessing_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all preprocessing jobs for current user."""
    jobs = db.query(PreprocessingJob).filter(PreprocessingJob.user_id == current_user.id).order_by(PreprocessingJob.created_at.desc()).all()
    
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
def get_preprocessing_job_status(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get status of a specific preprocessing job."""
    job = db.query(PreprocessingJob).filter(
        PreprocessingJob.id == job_id,
        PreprocessingJob.user_id == current_user.id
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


@router.get("/jobs/{job_id}/result", response_model=PreprocessingResult)
def get_preprocessing_result(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get preprocessing job result."""
    job = db.query(PreprocessingJob).filter(
        PreprocessingJob.id == job_id,
        PreprocessingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    artifact = job.artifact
    if not artifact:
        raise HTTPException(status_code=404, detail="Result artifact not found")
    
    # Load metrics
    import json
    with open(artifact.storage_path.replace(".npz", "_metrics.json"), 'r') as f:
        metrics = json.load(f)
    
    # Extract config hash from filename
    config_hash = artifact.metadata.get("config_hash", "")
    
    return PreprocessingResult(
        job_id=job.id,
        method=job.config.get("config", {}).get("preprocessing", {}).get("method", "unknown"),
        config_hash=config_hash,
        channels=artifact.metadata.get("channels", []),
        subjects=artifact.metadata.get("subjects", []),
        metrics=metrics,
        data_path=artifact.storage_path,
        scaler_path=artifact.storage_path.replace(".npz", "_scaler.pkl"),
        visualization_paths=[]
    )


@router.delete("/jobs/{job_id}")
def delete_preprocessing_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a preprocessing job."""
    job = db.query(PreprocessingJob).filter(
        PreprocessingJob.id == job_id,
        PreprocessingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    db.delete(job)
    db.commit()
    return {"message": "Job deleted"}