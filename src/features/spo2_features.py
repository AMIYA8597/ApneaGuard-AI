import numpy as np
from src.preprocessing.spo2 import DesaturationEvent
from typing import List

def compute_spo2_features(desaturation_events: List[DesaturationEvent], window_start_minute: int, fs: int, is_spo2_available: bool = True) -> dict:
    """
    Compute features from SpO2 desaturation events for a specific minute window.
    
    If SpO2 is not available for this record, returns NaNs with explicit flag handled by caller.
    """
    if not is_spo2_available:
        return {
            'spo2_dip_count': np.nan,
            'spo2_mean_dip_depth': np.nan,
            'spo2_mean_dip_duration': np.nan
        }
        
    samples_per_minute = fs * 60
    w_start = window_start_minute * samples_per_minute
    w_end = w_start + samples_per_minute
    
    # Events overlapping this window
    window_events = []
    for e in desaturation_events:
        # Overlap condition
        if e.start_idx < w_end and e.end_idx > w_start:
            window_events.append(e)
            
    if not window_events:
        return {
            'spo2_dip_count': 0.0,
            'spo2_mean_dip_depth': 0.0,
            'spo2_mean_dip_duration': 0.0
        }
        
    depths = [e.depth_pct for e in window_events]
    durations = [e.duration_samples for e in window_events]
    
    return {
        'spo2_dip_count': float(len(window_events)),
        'spo2_mean_dip_depth': float(np.mean(depths)),
        'spo2_mean_dip_duration': float(np.mean(durations))
    }
