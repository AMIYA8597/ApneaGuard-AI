from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api import schemas, db
from api.services import ml_pipeline

router = APIRouter()

@router.get("/{prediction_id}/explanation", response_model=schemas.ExplanationOut)
def get_explanation(prediction_id: int, db_session: Session = Depends(db.get_db)):
    try:
        exp = ml_pipeline.get_or_compute_explanation(prediction_id, db_session)
        return exp
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
