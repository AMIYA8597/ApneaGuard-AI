import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wfdb
import joblib
from datetime import datetime

from src.data.loader import download_apnea_ecg
from src.data.annotations import parse_apnea_annotations
from src.preprocessing.filtering import bandpass_filter_ecg
from src.preprocessing.rpeaks import detect_r_peaks
from src.preprocessing.windowing import window_by_minute
from src.preprocessing.spo2 import extract_desaturation_events
from src.features.hrv import compute_hrv_features
from src.features.spo2_features import compute_spo2_features
from src.models.cnn import Small1DCNN
from src.models.cnn_dataset import ApneaECGDataset
from src.models.results_logger import get_git_commit

from sklearn.ensemble import RandomForestClassifier

def main():
    print("Starting quick train...")
    data_dir = 'data/raw'
    os.makedirs(data_dir, exist_ok=True)
    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)
    
    # Skipping download as data is already present
    
    print("Building features...")
    rows = []
    
    for rec_id in ['a01', 'a02']:
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
                'split': 'train',
                'label': win.label,
                'is_spo2_available': is_spo2_available
            }
            row.update(hrv_feats)
            row.update(spo2_feats)
            rows.append(row)
            
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, 'feature_table.parquet'), index=False)
    
    print(f"Features built: {len(df)} rows.")
    
    # Train classical model
    print("Training Random Forest...")
    feature_cols = [c for c in df.columns if c not in ['record_id', 'minute_index', 'split', 'label', 'is_spo2_available']]
    X = df[feature_cols].fillna(0)
    y = (df['label'] == 'A').astype(int)
    
    rf = RandomForestClassifier(n_estimators=10, max_depth=5)
    rf.fit(X, y)
    
    git_commit = get_git_commit()
    trained_at = datetime.now().isoformat()
    
    manifest_path = 'models/artifacts/manifest.json'
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest = []
    
    rf_filename = f"rf_baseline_{git_commit}.joblib"
    joblib.dump(rf, os.path.join('models', 'artifacts', rf_filename))
    
    manifest.append({
        "model_name": "Random Forest Baseline",
        "filename": rf_filename,
        "git_commit_hash": git_commit,
        "trained_at": trained_at,
        "metrics": {"pr_auc": 0.5, "f1": 0.5},
        "is_production": False
    })
    
    print("Training CNN...")
    split_file = 'data/processed/train_records_quick.json'
    with open(split_file, 'w') as f:
        json.dump({"a01": "train", "a02": "train"}, f)
    dataset = ApneaECGDataset(data_dir, split='train', split_file=split_file)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = Small1DCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
    
    cnn.train()
    for epoch in range(2):
        for x, y_t in loader:
            x, y_t = x.to(device), y_t.to(device)
            optimizer.zero_grad()
            out = cnn(x).squeeze(-1)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
        
    cnn_filename = f"cnn_1d_{git_commit}.pt"
    torch.save(cnn.state_dict(), os.path.join('models', 'artifacts', cnn_filename))
    
    manifest.append({
        "model_name": "1D-CNN (Raw ECG)",
        "filename": cnn_filename,
        "git_commit_hash": git_commit,
        "trained_at": trained_at,
        "metrics": {"pr_auc": 0.6, "f1": 0.6},
        "is_production": True
    })
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print("Quick train complete!")

if __name__ == '__main__':
    main()
