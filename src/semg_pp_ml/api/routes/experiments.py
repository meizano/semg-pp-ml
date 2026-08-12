"""
Experiment routes for grid search and batch experiments.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_current_user, get_config
from ..schemas import (
    GridSearchRequest, GridSearchResponse, ExperimentResult, ExcludePair
)
from ..models.db import Experiment, User
from ...experiments.grid_search import run_grid_search_optimized, GridSearchResult
from ...config import ExperimentConfig, GridSearchConfig

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


async def run_experiment_job(
    job_id: int,
    config: ExperimentConfig,
    grid_config: GridSearchConfig,
    user_id: int,
    db: Session,
    skip_visualization: bool
):
    """Background task to run grid search experiment."""
    job = db.query(Experiment).filter(Experiment.id == job_id).first()
    if not job:
        return

    try:
        update_job_status(job_id, "running", 0.05, "Starting grid search")
        
        # Convert exclude pairs
        exclude_pairs = [pair.model_dump().values() for pair in grid_config.exclude_pairs]
        
        update_job_status(job_id, "running", 0.1, "Running experiments")
        
        # Run grid search
        result = run_grid_search_optimized(
            config,
            grid_config,
            skip_visualization=skip_visualization,
            parallel=grid_config.max_parallel > 1,
            max_workers=grid_config.max_parallel
        )
        
        update_job_status(job_id, "running", 0.9, "Saving results")
        
        # Update job
        job.status = "completed"
        job.results_artifact_id = None  # Would need to save results as artifact
        job.completed_at = datetime.utcnow()
        db.commit()
        
        update_job_status(job_id, "completed", 1.0, "Done")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
        update_job_status(job_id, "failed", 0, error=str(e))


@router.post("/grid-search", response_model=GridSearchResponse)
def start_grid_search(
    request: GridSearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    config: ExperimentConfig = Depends(get_config)
):
    """Start a grid search experiment."""
    # Build grid config
    grid_config = GridSearchConfig(
        preps=request.preps,
        models=request.models,
        exclude_pairs=[list(pair.model_dump().values()) for pair in request.exclude_pairs],
        skip_visualization=request.skip_visualization,
        max_parallel=request.max_parallel
    )
    
    # Create experiment record
    job = Experiment(
        user_id=current_user.id,
        name=f"grid_search_{uuid.uuid4().hex[:8]}",
        description=request.model_dump_json(),
        prep_methods=request.preps,
        models=request.models,
        exclude_pairs=[pair.model_dump() for pair in request.exclude_pairs],
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    job_status[job.id] = {"status": "pending", "progress": 0}
    
    background_tasks.add_task(
        run_experiment_job,
        job.id, config, grid_config, current_user.id, db, request.skip_visualization
    )
    
    return GridSearchResponse(
        experiment_id=job.id,
        status="pending",
        message="Grid search started"
    )


@router.get("/", response_model=List[ExperimentResult])
def list_experiments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all experiments for current user."""
    experiments = db.query(Experiment).filter(Experiment.user_id == current_user.id).order_by(Experiment.created_at.desc()).all()
    
    result = []
    for exp in experiments:
        status_info = job_status.get(exp.id, {"status": exp.status, "progress": 1.0 if exp.status == "completed" else 0})
        result.append(ExperimentResult(
            experiment_id=exp.id,
            name=exp.name,
            status=exp.status,
            total_combinations=len(exp.prep_methods) * len(exp.models),
            completed=0,  # Would need to track
            failed=0,
            results=[],
            best_by_metric={},
            total_time=0,
        ))
    return result


@router.get("/{experiment_id}", response_model=ExperimentResult)
def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get experiment details and results."""
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == current_user.id
    ).first()
    
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Load results from CSV if available
    import pandas as pd
    results = []
    best_by_metric = {}
    
    output_dir = Path("outputs") / "experiments" / str(experiment_id)
    csv_path = output_dir / "grid_search_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        results = df.to_dict('records')
        
        # Find best by metric
        ok_df = df[df['status'] == 'OK']
        for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc_roc_weighted']:
            if metric in ok_df.columns:
                best_idx = ok_df[metric].idxmax()
                best_by_metric[metric] = results[best_idx]
    
    return ExperimentResult(
        experiment_id=exp.id,
        name=exp.name,
        status=exp.status,
        total_combinations=len(exp.prep_methods) * len(exp.models),
        completed=len([r for r in results if r.get('status') == 'OK']),
        failed=len([r for r in results if r.get('status') == 'FAILED']),
        results=results,
        best_by_metric=best_by_metric,
        total_time=0,
        results_csv_path=str(csv_path) if csv_path.exists() else None
    )


@router.get("/{experiment_id}/status")
def get_experiment_status(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get experiment status."""
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == current_user.id
    ).first()
    
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    status_info = job_status.get(experiment_id, {"status": exp.status, "progress": 1.0 if exp.status == "completed" else 0})
    return {
        "experiment_id": exp.id,
        "status": exp.status,
        "progress": status_info.get("progress", 0),
        "current_step": status_info.get("current_step"),
        "error": status_info.get("error")
    }


@router.delete("/{experiment_id}")
def delete_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete an experiment."""
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == current_user.id
    ).first()
    
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    db.delete(exp)
    db.commit()
    return {"message": "Experiment deleted"}