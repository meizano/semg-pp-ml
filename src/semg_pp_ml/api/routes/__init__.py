"""
API routes package.
"""
from .auth import router as auth_router
from .preprocessing import router as preprocessing_router
from .models import router as models_router
from .experiments import router as experiments_router
from .artifacts import router as artifacts_router
from .websocket import router as websocket_router

__all__ = [
    "auth_router",
    "preprocessing_router",
    "models_router",
    "experiments_router",
    "artifacts_router",
    "websocket_router",
]