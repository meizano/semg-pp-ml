"""
Celery tasks for preprocessing jobs.
"""
from __future__ import annotations

from celery import shared_task
from ..workers.celery_app import celery_app


@shared_task(bind=True, name="preprocessing.run")
def run_preprocessing_task(self, job_id: int, config_dict: dict):
    """Celery task for preprocessing."""
    # This would run the preprocessing in a worker process
    # Implementation would load config, run pipeline, save results
    pass


@celery_app.task(name="preprocessing.cleanup")
def cleanup_old_preprocessing(max_age_days: int = 30):
    """Cleanup old preprocessing artifacts."""
    pass