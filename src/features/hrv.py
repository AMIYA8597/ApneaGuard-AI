import numpy as np
import logging
from scipy.interpolate import interp1d
from scipy.signal import welch

logger = logging.getLogger(__name__)

def compute_hrv_features(r_peaks: np.ndarray, fs: int) -> dict:
    """
    Compute time-domain and frequency-domain HRV features from R-peaks.
    Returns NaN for all features if there are too few peaks.
    """
    if len(r_peaks) < 3:
        logger.warning(f"Too few R-peaks ({len(r_peaks)}) in window to compute meaningful HRV. Returning NaNs.")
        return {
            'SDNN': np.nan,
            'RMSSD': np.nan,
            'pNN50': np.nan,
            'LF_HF_ratio': np.nan
        }
        
    # RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / fs * 1000
    
    # Time-domain
    sdnn = np.std(rr_intervals, ddof=1)
    
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff**2))
    
    nn50 = np.sum(np.abs(rr_diff) > 50)
    pnn50 = (nn50 / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0.0
    
    # Frequency-domain (LF/HF)
    # Interpolate RR intervals to a regular grid (e.g., 4Hz)
    t_rr = np.cumsum(rr_intervals) / 1000.0  # seconds
    # Make sure we have enough duration
    if t_rr[-1] - t_rr[0] < 5:
        # Too short for meaningful frequency analysis
        lf_hf_ratio = np.nan
    else:
        try:
            fs_interp = 4.0
            t_interp = np.arange(t_rr[0], t_rr[-1], 1/fs_interp)
            f_interp = interp1d(t_rr, rr_intervals, kind='cubic', fill_value='extrapolate')
            rr_interp = f_interp(t_interp)
            
            # Welch's method
            # Window size = 256 or length of signal if shorter
            nperseg = min(256, len(rr_interp))
            if nperseg < 4:
                lf_hf_ratio = np.nan
            else:
                freqs, psd = welch(rr_interp, fs=fs_interp, nperseg=nperseg)
                
                # LF: 0.04 - 0.15 Hz, HF: 0.15 - 0.4 Hz
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                
                lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
                hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
                
                if hf_power > 0:
                    lf_hf_ratio = lf_power / hf_power
                else:
                    lf_hf_ratio = np.nan
        except Exception as e:
            logger.debug(f"Frequency domain feature extraction failed: {e}")
            lf_hf_ratio = np.nan
            
    return {
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'LF_HF_ratio': lf_hf_ratio
    }
