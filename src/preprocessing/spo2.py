import numpy as np
import logging
from dataclasses import dataclass
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class DesaturationEvent:
    start_idx: int
    end_idx: int
    duration_samples: int
    depth_pct: float

def extract_desaturation_events(spo2_signal: np.ndarray, fs: int, dip_threshold_pct: float = 3.0) -> List[DesaturationEvent]:
    """
    Extract SpO2 desaturation events. 
    A desaturation event is defined as a drop of at least dip_threshold_pct 
    from a local rolling baseline.
    """
    if spo2_signal is None or len(spo2_signal) == 0:
        logger.warning("Record lacks an SpO2 channel. Skipping SpO2 processing.")
        return []
        
    # Baseline estimation: 3-minute rolling max
    # Using a 3-minute window to find the recent "stable" baseline
    window_samples = fs * 60 * 3
    
    # Use pandas rolling for efficient rolling max
    spo2_series = pd.Series(spo2_signal)
    baseline = spo2_series.rolling(window=window_samples, min_periods=1).max().values
    
    # Identify dips
    dip_signal = baseline - spo2_signal
    is_dipping = dip_signal >= dip_threshold_pct
    
    events = []
    in_event = False
    start_idx = 0
    max_dip = 0.0
    
    for i, dipping in enumerate(is_dipping):
        if dipping and not in_event:
            in_event = True
            start_idx = i
            max_dip = dip_signal[i]
        elif dipping and in_event:
            if dip_signal[i] > max_dip:
                max_dip = dip_signal[i]
        elif not dipping and in_event:
            in_event = False
            end_idx = i
            events.append(DesaturationEvent(
                start_idx=start_idx,
                end_idx=end_idx,
                duration_samples=end_idx - start_idx,
                depth_pct=max_dip
            ))
            
    # Handle event finishing at end of signal
    if in_event:
        end_idx = len(spo2_signal)
        events.append(DesaturationEvent(
            start_idx=start_idx,
            end_idx=end_idx,
            duration_samples=end_idx - start_idx,
            depth_pct=max_dip
        ))
        
    return events
