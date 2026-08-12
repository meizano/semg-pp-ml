"""
Celery tasks for training jobs.
"""
from __future__ import annotations

from celery import shared_task
from ..workers.celery_app import celery_app


@shared_task(bind=True, name="training.run")
def run_training_task(self, job_id: int, config_dict: dict):
    """Celery task for model training."""
    pass


@celery_app.task(name="training.cleanup")
def cleanup_old_models(max_age_days: int = 30):
    """Cleanup old model artifacts."""
    pass