from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from api import schemas, models, db

router = APIRouter()

@router.get("/{recording_id}/predictions", response_model=List[schemas.PredictionOut])
def get_predictions(recording_id: str, db_session: Session = Depends(db.get_db)):
    preds = db_session.query(models.Prediction).filter(models.Prediction.recording_id == recording_id).all()
    return preds
