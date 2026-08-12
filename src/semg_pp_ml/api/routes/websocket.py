"""
WebSocket routes for real-time updates and streaming.
"""
from __future__ import annotations

import json
import asyncio
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_current_user_ws
from ..services.websocket_manager import manager

router = APIRouter()


@router.websocket("/jobs/{job_id}")
async def job_progress_websocket(
    websocket: WebSocket,
    job_id: str,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    """WebSocket for job progress updates."""
    user = await get_current_user_ws(websocket, token, db)
    if not user:
        return
    
    await manager.connect(websocket, f"job_{job_id}")
    try:
        # Send current status if available
        from ..routes.preprocessing import job_status as prep_status
        from ..routes.models import job_status as model_status
        from ..routes.experiments import job_status as exp_status
        
        for status_dict in [prep_status, model_status, exp_status]:
            if int(job_id) in status_dict:
                await websocket.send_json({
                    "type": "job_update",
                    "job_id": int(job_id),
                    **status_dict[int(job_id)]
                })
                break
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, f"job_{job_id}")


@router.websocket("/stream/{session_id}")
async def emg_stream_websocket(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    """WebSocket for live sEMG data streaming."""
    user = await get_current_user_ws(websocket, token, db)
    if not user:
        return
    
    await manager.connect(websocket, f"stream_{session_id}")
    try:
        while True:
            # Receive data from sender (Arduino/device) or broadcast to viewers
            data = await websocket.receive_json()
            
            # Broadcast to all viewers of this stream
            await manager.send_streaming_data(session_id, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket, f"stream_{session_id}")


@router.websocket("/notifications")
async def notifications_websocket(
    websocket: WebSocket,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    """WebSocket for general notifications."""
    user = await get_current_user_ws(websocket, token, db)
    if not user:
        return
    
    await manager.connect(websocket, f"notifications_{user.id}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, f"notifications_{user.id}")