"""
WebSocket connection manager for real-time updates.
"""
from __future__ import annotations

from typing import Dict, Set
from fastapi import WebSocket
import asyncio


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)
    
    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
            if not self.active_connections[channel]:
                del self.active_connections[channel]
    
    async def broadcast(self, channel: str, message: dict):
        if channel in self.active_connections:
            dead = set()
            for ws in self.active_connections[channel]:
                try:
                    await ws.send_json(message)
                except:
                    dead.add(ws)
            for ws in dead:
                self.active_connections[channel].discard(ws)
    
    async def send_job_update(self, job_id: str, status: str, progress: float, metrics: dict = None):
        """Send job progress update to subscribers."""
        msg = {
            "type": "job_update",
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "metrics": metrics
        }
        await self.broadcast(f"job_{job_id}", msg)
    
    async def send_streaming_data(self, session_id: str, data: dict):
        """Broadcast live sEMG data to stream viewers."""
        msg = {"type": "emg_data", "session_id": session_id, **data}
        await self.broadcast(f"stream_{session_id}", msg)
    
    async def send_notification(self, user_id: int, message: dict):
        """Send notification to specific user."""
        await self.broadcast(f"notifications_{user_id}", {"type": "notification", **message})


manager = ConnectionManager()