import pytest
import numpy as np
import pandas as pd
from src.preprocessing.rpeaks import detect_r_peaks
from src.preprocessing.windowing import window_by_minute

def test_synthetic_ecg_rpeak_detection():
    # Generate a clean synthetic ECG-like signal with a known heart rate
    fs = 100
    duration_sec = 60
    bpm = 60
    # exactly 60 beats in 60 seconds
    
    t = np.arange(0, duration_sec, 1/fs)
    signal = np.zeros_like(t)
    
    # Place a 'spike' (R-peak) exactly every 1 second
    peak_indices = np.arange(0, len(t), fs)
    signal[peak_indices] = 1.0
    
    # Smooth the spike slightly so it resembles an ECG peak for NeuroKit2
    # Simple convolution
    kernel = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
    signal = np.convolve(signal, kernel, mode='same')
    
    # Test
    detected_peaks = detect_r_peaks(signal, fs)
    
    # We expect 60 peaks. NeuroKit2 should easily detect these distinct peaks
    # Allow a very small tolerance (e.g. ±1 beat) for edge effects
    assert abs(len(detected_peaks) - bpm) <= 1
    
def test_windowing_alignment():
    # Spot-check windowing alignment
    fs = 100
    duration_min = 5
    total_samples = duration_min * 60 * fs
    
    raw_ecg = np.zeros(total_samples)
    filtered_ecg = np.zeros(total_samples)
    r_peaks = np.array([100, 200, 7000]) # Example peaks
    
    # Create annotations dataframe for 5 minutes
    annotations = pd.DataFrame({
        'minute_index': [0, 1, 2, 3, 4],
        'label': ['normal', 'apnea', 'normal', 'apnea', 'normal']
    })
    
    windows = window_by_minute('mock', raw_ecg, filtered_ecg, r_peaks, fs, annotations)
    
    assert len(windows) == 5
    assert windows[0].minute_index == 0
    assert windows[0].label == 'normal'
    assert len(windows[0].raw_ecg) == 60 * fs
    
    # R-peak at index 7000 is in minute 1 (6000 to 12000). 7000 - 6000 = 1000 relative index
    assert 1000 in windows[1].r_peaks
    assert len(windows[1].r_peaks) == 1
    assert len(windows[0].r_peaks) == 2 # 100 and 200 relative indices
