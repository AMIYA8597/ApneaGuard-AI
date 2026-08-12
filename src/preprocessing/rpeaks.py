import numpy as np
import neurokit2 as nk

def detect_r_peaks(filtered_ecg: np.ndarray, fs: int) -> np.ndarray:
    """
    Detect R-peaks in a filtered ECG signal using NeuroKit2.
    
    Tradeoff Note:
    We use NeuroKit2's built-in peak detection rather than hand-rolling 
    the classic Pan-Tompkins algorithm from scratch. While hand-rolling Pan-Tompkins 
    would offer deep educational value and finer-grained algorithmic control as a stretch goal,
    NeuroKit2 provides a heavily tested, edge-case-resilient, production-ready implementation 
    that lets us focus on the higher-level apnea detection architecture without getting bogged
    down in DSP edge cases.
    """
    # Returns a tuple: signals (DataFrame) and info (Dict with 'ECG_R_Peaks' containing indices)
    _, info = nk.ecg_peaks(filtered_ecg, sampling_rate=fs)
    r_peaks = np.array(info['ECG_R_Peaks'], dtype=int)
    return r_peaks
