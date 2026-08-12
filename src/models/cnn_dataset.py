import os
import torch
import wfdb
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.preprocessing.filtering import bandpass_filter_ecg
from src.data.annotations import parse_apnea_annotations

class ApneaECGDataset(Dataset):
    """
    PyTorch Dataset loading raw filtered per-minute ECG windows.
    Re-uses the EXACT subject-level split from split_assignment.json to guarantee 
    honest comparison against classical models.
    """
    def __init__(self, data_dir: str, split: str = 'train', split_file: str = 'data/processed/split_assignment.json'):
        self.data_dir = data_dir
        
        with open(split_file, 'r') as f:
            all_splits = json.load(f)
            
        # Get record IDs belonging to the requested split
        self.record_ids = [rec_id for rec_id, s in all_splits.items() if s == split]
        
        self.windows = []
        self.labels = []
        
        # We pre-load the windows here for simplicity (dataset is small enough to fit in RAM)
        for rec_id in self.record_ids:
            rec_path = os.path.join(data_dir, rec_id)
            if not os.path.exists(rec_path + '.dat'):
                continue
                
            record = wfdb.rdrecord(rec_path)
            fs = record.fs
            sig_name = record.sig_name
            ecg_idx = sig_name.index('ECG') if 'ECG' in sig_name else 0
            raw_ecg = record.p_signal[:, ecg_idx]
            
            filtered_ecg = bandpass_filter_ecg(raw_ecg, fs)
            
            annotations = parse_apnea_annotations(rec_id, data_dir)
            if annotations.empty:
                continue
                
            samples_per_minute = fs * 60
            num_minutes = min(len(filtered_ecg) // samples_per_minute, len(annotations))
            
            for min_idx in range(num_minutes):
                start_idx = min_idx * samples_per_minute
                end_idx = start_idx + samples_per_minute
                
                # Normalize window
                window_sig = filtered_ecg[start_idx:end_idx]
                if np.std(window_sig) > 0:
                    window_sig = (window_sig - np.mean(window_sig)) / np.std(window_sig)
                else:
                    window_sig = window_sig - np.mean(window_sig)
                    
                label_str = annotations.iloc[min_idx]['label']
                label_bin = 1 if label_str == 'apnea' else 0
                
                self.windows.append(window_sig)
                self.labels.append(label_bin)
                
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        # Return as float tensor with shape [Channels, Length] -> [1, 6000]
        x = torch.tensor(self.windows[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
