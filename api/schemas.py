from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

class RecordingBase(BaseModel):
    id: str = Field(..., description="The ID of the recording, e.g. a01")
    
class RecordingCreate(RecordingBase):
    pass

class PredictionBase(BaseModel):
    minute_index: int
    is_apnea: bool
    probability: float
    model_version_id: str

class PredictionOut(PredictionBase):
    id: int
    
    model_config = ConfigDict(from_attributes=True)

class SeverityScoreBase(BaseModel):
    model_version_id: str
    duration_hours: float
    ahi: float
    severity_band: str

class SeverityScoreOut(SeverityScoreBase):
    id: int
    
    model_config = ConfigDict(from_attributes=True)

class RecordingOut(RecordingBase):
    status: str
    duration_minutes: Optional[int]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class RecordingDetailOut(RecordingOut):
    severity: Optional[SeverityScoreOut] = None
    predictions_count: int = 0
    
    model_config = ConfigDict(from_attributes=True)

class ExplanationOut(BaseModel):
    id: int
    prediction_id: int
    method: str
    plot_path: str
    top_features: Optional[Dict[str, float]] = None
    
    model_config = ConfigDict(from_attributes=True)
