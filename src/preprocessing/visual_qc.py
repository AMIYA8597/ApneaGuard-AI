import os
import sys

# Ensure src module is reachable when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.filtering import bandpass_filter_ecg
from src.preprocessing.rpeaks import detect_r_peaks

def main():
    data_dir = 'data/raw'
    out_dir = 'docs/visual_qc'
    os.makedirs(out_dir, exist_ok=True)
    
    # Pick 3 annotated records for QC
    records_to_check = ['a01', 'b01', 'c01']
    
    for rec_id in records_to_check:
        rec_path = os.path.join(data_dir, rec_id)
        if not os.path.exists(rec_path + '.dat'):
            print(f"Record {rec_id} not found in {data_dir}. Run Phase 1 first.")
            continue
            
        record = wfdb.rdrecord(rec_path)
        fs = record.fs
        
        # Typically the first channel is ECG
        sig_name = record.sig_name
        ecg_idx = 0
        if 'ECG' in sig_name:
            ecg_idx = sig_name.index('ECG')
            
        raw_ecg = record.p_signal[:, ecg_idx]
        
        # Take a short 10-second segment for clear plotting
        segment_len = fs * 10 
        raw_seg = raw_ecg[:segment_len]
        
        filtered_seg = bandpass_filter_ecg(raw_seg, fs)
        r_peaks = detect_r_peaks(filtered_seg, fs)
        
        # Plot
        plt.figure(figsize=(15, 5))
        time_axis = np.arange(len(filtered_seg)) / fs
        
        plt.plot(time_axis, raw_seg, label='Raw ECG', alpha=0.5, color='gray')
        plt.plot(time_axis, filtered_seg, label='Filtered ECG', color='blue')
        plt.plot(r_peaks / fs, filtered_seg[r_peaks], 'ro', markersize=8, label='Detected R-peaks')
        
        plt.title(f'Record {rec_id} - 10 Second ECG Segment QC')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        out_file = os.path.join(out_dir, f'qc_{rec_id}.png')
        plt.savefig(out_file)
        plt.close()
        
        print(f"Saved visual QC plot for {rec_id} to {out_file}")

if __name__ == '__main__':
    main()
