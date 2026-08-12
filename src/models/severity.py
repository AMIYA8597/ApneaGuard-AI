from dataclasses import dataclass

@dataclass
class WindowPrediction:
    minute_index: int
    is_apnea: bool
    probability: float

@dataclass
class SeverityResult:
    total_events: int
    duration_hours: float
    ahi: float
    severity_band: str

def compute_severity(predictions: list[WindowPrediction], recording_duration_minutes: int) -> SeverityResult:
    """
    Computes the AHI (Apnea-Hypopnea Index) and the corresponding AASM severity band.
    
    Args:
        predictions: List of WindowPrediction objects for the recording.
        recording_duration_minutes: The actual recording duration in minutes. 
                                    (Do not assume a fixed 8-hour night)
                                    
    Returns:
        SeverityResult containing the computed metrics.
    """
    if recording_duration_minutes <= 0:
        raise ValueError("Recording duration must be strictly positive.")
        
    duration_hours = recording_duration_minutes / 60.0
    
    total_events = sum(1 for p in predictions if p.is_apnea)
    
    ahi = total_events / duration_hours
    
    if ahi < 5:
        band = "normal"
    elif ahi < 15:
        band = "mild"
    elif ahi < 30:
        band = "moderate"
    else:
        band = "severe"
        
    return SeverityResult(
        total_events=total_events,
        duration_hours=duration_hours,
        ahi=ahi,
        severity_band=band
    )
