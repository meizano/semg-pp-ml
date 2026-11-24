"""
Artifact management routes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_current_user
from ..schemas import ArtifactResponse, ArtifactListResponse, ArtifactType
from ..models.db import Artifact, User

router = APIRouter()


@router.get("/", response_model=ArtifactListResponse)
def list_artifacts(
    artifact_type: Optional[ArtifactType] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List artifacts with pagination and filtering."""
    query = db.query(Artifact)
    
    if artifact_type:
        query = query.filter(Artifact.type == artifact_type.value)
    
    total = query.count()
    artifacts = query.order_by(Artifact.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    
    return ArtifactListResponse(
        artifacts=[
            ArtifactResponse(
                id=a.id,
                name=a.name,
                type=a.type,
                size_bytes=a.size_bytes,
                mime_type=a.mime_type,
                metadata=a.meta,
                created_at=a.created_at,
                download_url=f"/api/v1/artifacts/{a.id}/download"
            )
            for a in artifacts
        ],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{artifact_id}", response_model=ArtifactResponse)
def get_artifact(
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get artifact metadata."""
    artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return ArtifactResponse(
        id=artifact.id,
        name=artifact.name,
        type=artifact.type,
        size_bytes=artifact.size_bytes,
        mime_type=artifact.mime_type,
        metadata=artifact.meta,
        created_at=artifact.created_at,
        download_url=f"/api/v1/artifacts/{artifact.id}/download"
    )


@router.get("/{artifact_id}/download")
def download_artifact(
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download artifact file."""
    artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    file_path = Path(artifact.storage_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        path=file_path,
        filename=artifact.name,
        media_type=artifact.mime_type
    )


@router.delete("/{artifact_id}")
def delete_artifact(
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete artifact (metadata only, file remains)."""
    artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    db.delete(artifact)
    db.commit()
    return {"message": "Artifact metadata deleted"}


@router.post("/{artifact_id}/cleanup")
def cleanup_artifact_file(
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete artifact file from disk."""
    artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    file_path = Path(artifact.storage_path)
    if file_path.exists():
        file_path.unlink()
    
    return {"message": "Artifact file deleted"}