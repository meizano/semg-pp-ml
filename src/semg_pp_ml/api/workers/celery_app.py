"""
Celery configuration for background tasks.
"""
from __future__ import annotations

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "semg_pp_ml",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "semg_pp_ml.api.workers.preprocessing_tasks",
        "semg_pp_ml.api.workers.training_tasks",
        "semg_pp_ml.api.workers.experiment_tasks",
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
)