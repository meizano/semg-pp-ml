"""
FastAPI application for sEMG Preprocessing and ML Platform.
"""
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .dependencies import get_db, get_current_user
from .routes import auth, preprocessing, models, experiments, artifacts, websocket
from .services.websocket_manager import manager
from ..config import ExperimentConfig, load_config
from ..utils.system import get_system_info, print_system_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting sEMG Preprocessing & ML Platform...")

    # Print system info
    sys_info = get_system_info()
    print_system_info(sys_info)

    # Load default config
    config_path = Path(os.getenv("CONFIG_PATH", "config/default.yaml"))
    if config_path.exists():
        app.state.config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        app.state.config = ExperimentConfig()
        logger.warning(f"Config not found at {config_path}, using defaults")

    # Create output directories
    output_dir = Path(app.state.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "preprocessing").mkdir(exist_ok=True)

    logger.info("Startup complete")
    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="sEMG Preprocessing & ML Platform",
    description="API for NinaPro DB8 sEMG gesture classification experiments",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(preprocessing.router, prefix="/api/v1/preprocessing", tags=["Preprocessing"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["Experiments"])
app.include_router(artifacts.router, prefix="/api/v1/artifacts", tags=["Artifacts"])
app.include_router(websocket.router, prefix="/api/v1/ws", tags=["WebSocket"])

# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

# System info
@app.get("/api/v1/system", tags=["System"])
async def system_info(current_user=Depends(get_current_user)):
    sys_info = get_system_info()
    return sys_info.to_dict()

# Config endpoints
@app.get("/api/v1/config", tags=["Configuration"])
async def get_config(current_user=Depends(get_current_user)):
    return app.state.config.model_dump(mode='json')

@app.post("/api/v1/config", tags=["Configuration"])
async def update_config(config: ExperimentConfig, current_user=Depends(get_current_user)):
    app.state.config = config
    return {"message": "Configuration updated", "config": config.model_dump(mode='json')}

# Serve frontend (if built)
frontend_path = Path(__file__).parent.parent.parent.parent / "web" / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("semg_pp_ml.api.main:app", host="0.0.0.0", port=8000, reload=True)