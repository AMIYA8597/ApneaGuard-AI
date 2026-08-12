from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import datetime
from api import schemas, models, db
from api.services import ml_pipeline

router = APIRouter()

@router.post("/", response_model=schemas.RecordingOut)
def create_recording(recording: schemas.RecordingCreate, db_session: Session = Depends(db.get_db)):
    db_rec = db_session.query(models.Recording).filter(models.Recording.id == recording.id).first()
    if db_rec:
        raise HTTPException(status_code=400, detail="Recording already exists")
    
    new_rec = models.Recording(
        id=recording.id,
        status="ingested",
        created_at=datetime.datetime.utcnow()
    )
    db_session.add(new_rec)
    db_session.commit()
    db_session.refresh(new_rec)
    return new_rec

@router.get("/", response_model=List[schemas.RecordingOut])
def list_recordings(db_session: Session = Depends(db.get_db)):
    return db_session.query(models.Recording).all()

@router.get("/{recording_id}", response_model=schemas.RecordingDetailOut)
def get_recording(recording_id: str, db_session: Session = Depends(db.get_db)):
    db_rec = db_session.query(models.Recording).filter(models.Recording.id == recording_id).first()
    if not db_rec:
        raise HTTPException(status_code=404, detail="Recording not found")
        
    predictions_count = db_session.query(models.Prediction).filter(models.Prediction.recording_id == recording_id).count()
    
    # We construct the detail response
    res = schemas.RecordingDetailOut.model_validate(db_rec)
    res.predictions_count = predictions_count
    return res

@router.post("/{recording_id}/predict", response_model=List[schemas.PredictionOut])
def predict_recording(recording_id: str, model_version: str = "cnn", db_session: Session = Depends(db.get_db)):
    if model_version not in ["cnn", "classical"]:
        raise HTTPException(status_code=422, detail="Invalid model version. Must be 'cnn' or 'classical'.")
        
    db_rec = db_session.query(models.Recording).filter(models.Recording.id == recording_id).first()
    if not db_rec:
        raise HTTPException(status_code=404, detail="Recording not found")
        
    try:
        preds = ml_pipeline.run_predictions(recording_id, db_session, model_version=model_version)
        return preds
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
