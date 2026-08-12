import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# We use the real test DB (or sqlite memory) to check constraints
from api.db import engine, Base, SessionLocal
from api.models import ModelVersion, Prediction, Recording

# Ensure tables are created for tests
Base.metadata.create_all(bind=engine)

def test_model_version_commit_hash_not_null():
    """Test that git_commit_hash cannot be null at the schema level."""
    session = SessionLocal()
    
    # Try inserting without git_commit_hash
    mv = ModelVersion(id="test_mv_null", description="Should fail")
    session.add(mv)
    
    with pytest.raises(IntegrityError):
        session.commit()
        
    session.rollback()

def test_prediction_model_version_id_not_null():
    """Test that model_version_id cannot be null at the schema level."""
    session = SessionLocal()
    
    rec = Recording(id="test_rec_schema", duration_minutes=10, split_role="train")
    session.add(rec)
    session.commit()
    
    # Try inserting Prediction without model_version_id
    pred = Prediction(
        recording_id="test_rec_schema",
        minute_index=0,
        is_apnea=False,
        probability=0.1
    )
    session.add(pred)
    
    with pytest.raises(IntegrityError):
        session.commit()
        
    session.rollback()
