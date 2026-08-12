"""
Database models using SQLAlchemy.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Enum, JSON, Boolean,
    create_engine
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool
import os


Base = declarative_base()


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(Enum(UserRole), default=UserRole.RESEARCHER, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    preprocessing_jobs = relationship("PreprocessingJob", back_populates="user")
    training_jobs = relationship("TrainingJob", back_populates="user")
    experiments = relationship("Experiment", back_populates="user")


class PreprocessingJob(Base):
    __tablename__ = "preprocessing_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255))
    config = Column(JSON, nullable=False)  # Full preprocessing config
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    input_data_hash = Column(String(64))
    output_artifact_id = Column(Integer, ForeignKey("artifacts.id"), nullable=True)
    metrics = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="preprocessing_jobs")
    artifact = relationship("Artifact", foreign_keys=[output_artifact_id])


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255))
    model_type = Column(String(100), nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    preprocessing_job_id = Column(Integer, ForeignKey("preprocessing_jobs.id"), nullable=True)
    config = Column(JSON, nullable=False)  # Full training config
    status = Column(String(50), default="pending")
    mlflow_run_id = Column(String(255), nullable=True)
    metrics = Column(JSON, nullable=True)
    model_artifact_id = Column(Integer, ForeignKey("artifacts.id"), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="training_jobs")
    preprocessing_job = relationship("PreprocessingJob")
    model_artifact = relationship("Artifact", foreign_keys=[model_artifact_id])


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    prep_methods = Column(JSON, nullable=False)  # List of preprocessing configs
    models = Column(JSON, nullable=False)  # List of model types + hyperparams
    exclude_pairs = Column(JSON, nullable=True)
    status = Column(String(50), default="pending")
    results_artifact_id = Column(Integer, ForeignKey("artifacts.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="experiments")


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # preprocessed_data, model, plot, report, tflite
    storage_path = Column(String(512), nullable=False)  # MinIO path or local path
    size_bytes = Column(Integer)
    mime_type = Column(String(100))
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://semg:semgpass@localhost:5432/semg_ml")

engine = create_engine(DATABASE_URL, poolclass=NullPool)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)