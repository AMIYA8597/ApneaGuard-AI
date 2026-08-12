import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter_ecg(signal: np.ndarray, fs: int, low_hz: float = 0.5, high_hz: float = 40.0) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the ECG signal using filtfilt.
    
    Why zero-phase (filtfilt) matters here:
    Using a standard causal filter (like lfilter) introduces a frequency-dependent phase shift, 
    meaning different frequency components of the QRS complex would be delayed by different amounts.
    This phase shift would physically misalign the detected R-peak timing against the per-minute 
    annotation boundaries, causing temporal leakage or misclassification. 
    `filtfilt` filters forward and backward, cancelling out the phase shift resulting in zero-phase delay.
    """
    nyquist = 0.5 * fs
    low = low_hz / nyquist
    high = high_hz / nyquist
    
    # 3rd-order Butterworth filter
    b, a = butter(3, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, signal)
    
    return filtered_ecg
