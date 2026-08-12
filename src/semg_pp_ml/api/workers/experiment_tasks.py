"""
Celery tasks for experiment jobs.
"""
from __future__ import annotations

from celery import shared_task
from ..workers.celery_app import celery_app


@shared_task(bind=True, name="experiment.grid_search")
def run_grid_search_task(self, experiment_id: int, config_dict: dict):
    """Celery task for grid search experiment."""
    pass


@celery_app.task(name="experiment.cleanup")
def cleanup_old_experiments(max_age_days: int = 30):
    """Cleanup old experiment artifacts."""
    pass