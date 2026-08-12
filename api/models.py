from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
import datetime
from .db import Base

class Recording(Base):
    __tablename__ = 'recordings'
    id = Column(String, primary_key=True, index=True) # e.g. "a01"
    status = Column(String, default="ingested")
    duration_minutes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    split_role = Column(String, nullable=False, default="unknown")
    
    predictions = relationship("Prediction", back_populates="recording")
    severity = relationship("SeverityScore", back_populates="recording", uselist=False)

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    id = Column(String, primary_key=True) # e.g. "cnn_v1"
    git_commit_hash = Column(String, nullable=False)
    description = Column(String, nullable=True)
    
    predictions = relationship("Prediction", back_populates="model_version")

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recording_id = Column(String, ForeignKey("recordings.id"), nullable=False)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)
    minute_index = Column(Integer, nullable=False)
    is_apnea = Column(Boolean, nullable=False)
    probability = Column(Float, nullable=False)
    
    recording = relationship("Recording", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")
    explanation = relationship("Explanation", back_populates="prediction", uselist=False)

class SeverityScore(Base):
    __tablename__ = 'severity_scores'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recording_id = Column(String, ForeignKey("recordings.id"), unique=True, nullable=False)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)
    duration_hours = Column(Float, nullable=False)
    ahi = Column(Float, nullable=False)
    severity_band = Column(String, nullable=False) # Normal, Mild, Moderate, Severe
    
    recording = relationship("Recording", back_populates="severity")
    model_version = relationship("ModelVersion")

class Explanation(Base):
    __tablename__ = 'explanations'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    method = Column(String, nullable=False) # "shap" or "saliency"
    plot_path = Column(String, nullable=False)
    top_features = Column(JSON, nullable=True)
    
    prediction = relationship("Prediction", back_populates="explanation")
