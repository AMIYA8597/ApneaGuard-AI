from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api import schemas, db
from api.services import ml_pipeline

router = APIRouter()

@router.get("/{recording_id}/severity", response_model=schemas.SeverityScoreOut)
def get_severity(recording_id: str, db_session: Session = Depends(db.get_db)):
    try:
        score = ml_pipeline.get_or_compute_severity(recording_id, db_session)
        return score
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
