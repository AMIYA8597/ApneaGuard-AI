import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SignalWindow:
    minute_index: int
    raw_ecg: np.ndarray
    filtered_ecg: np.ndarray
    r_peaks: np.ndarray
    label: Optional[str]

def window_by_minute(record_id: str, raw_ecg: np.ndarray, filtered_ecg: np.ndarray, 
                     r_peaks: np.ndarray, fs: int, annotations: Optional[pd.DataFrame] = None) -> List[SignalWindow]:
    """
    Window the ECG signal and R-peaks into 1-minute non-overlapping segments.
    Aligns exactly to the annotation minute boundaries to ensure label correspondence.
    """
    windows = []
    samples_per_minute = fs * 60
    
    # Total possible full minutes based on signal length
    total_minutes = len(raw_ecg) // samples_per_minute
    
    # Only iterate up to what's available in annotations (if provided)
    num_minutes = min(total_minutes, len(annotations)) if annotations is not None else total_minutes
    
    for min_idx in range(num_minutes):
        start_idx = min_idx * samples_per_minute
        end_idx = start_idx + samples_per_minute
        
        # Segment signals
        raw_seg = raw_ecg[start_idx:end_idx]
        filt_seg = filtered_ecg[start_idx:end_idx]
        
        # Segment R-peaks (adjust indices relative to the start of the window)
        mask = (r_peaks >= start_idx) & (r_peaks < end_idx)
        peaks_in_window = r_peaks[mask] - start_idx
        
        label = annotations.iloc[min_idx]['label'] if annotations is not None else None
        
        window = SignalWindow(
            minute_index=min_idx,
            raw_ecg=raw_seg,
            filtered_ecg=filt_seg,
            r_peaks=peaks_in_window,
            label=label
        )
        windows.append(window)
        
    return windows
