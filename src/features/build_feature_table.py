import os
import sys

# Ensure src module is reachable when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.split import assign_split
from src.data.loader import download_apnea_ecg
from src.data.annotations import parse_apnea_annotations
from src.preprocessing.filtering import bandpass_filter_ecg
from src.preprocessing.rpeaks import detect_r_peaks
from src.preprocessing.windowing import window_by_minute
from src.preprocessing.spo2 import extract_desaturation_events
from src.features.hrv import compute_hrv_features
from src.features.spo2_features import compute_spo2_features
import logging

logging.basicConfig(level=logging.WARNING)

def main():
    data_dir = 'data/raw'
    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)
    
    record_ids = download_apnea_ecg(data_dir)
    splits = assign_split(record_ids)
    
    rows = []
    
    for rec_id, split in tqdm(splits.items(), desc="Building feature table"):
        if split == 'holdout':
            continue
            
        rec_path = os.path.join(data_dir, rec_id)
        if not os.path.exists(rec_path + '.dat'):
            continue
            
        record = wfdb.rdrecord(rec_path)
        fs = record.fs
        sig_name = record.sig_name
        
        ecg_idx = sig_name.index('ECG') if 'ECG' in sig_name else 0
        raw_ecg = record.p_signal[:, ecg_idx]
        
        is_spo2_available = 'SpO2' in sig_name
        spo2_events = []
        if is_spo2_available:
            spo2_idx = sig_name.index('SpO2')
            spo2_signal = record.p_signal[:, spo2_idx]
            spo2_events = extract_desaturation_events(spo2_signal, fs)
            
        annotations = parse_apnea_annotations(rec_id, data_dir)
        if annotations.empty:
            continue
            
        filtered_ecg = bandpass_filter_ecg(raw_ecg, fs)
        r_peaks = detect_r_peaks(filtered_ecg, fs)
        
        windows = window_by_minute(rec_id, raw_ecg, filtered_ecg, r_peaks, fs, annotations)
        
        for win in windows:
            hrv_feats = compute_hrv_features(win.r_peaks, fs)
            spo2_feats = compute_spo2_features(spo2_events, win.minute_index, fs, is_spo2_available)
            
            row = {
                'record_id': rec_id,
                'minute_index': win.minute_index,
                'split': split,
                'label': win.label,
                'is_spo2_available': is_spo2_available
            }
            row.update(hrv_feats)
            row.update(spo2_feats)
            
            rows.append(row)
            
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, 'feature_table.parquet')
    df.to_parquet(out_path, index=False)
    print(f"\nSaved feature table with {len(df)} rows to {out_path}")

if __name__ == '__main__':
    main()
