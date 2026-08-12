import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from src.models.cnn_dataset import ApneaECGDataset
from src.models.cnn import Small1DCNN
from src.models.cross_validation import subject_level_kfold
from src.models.results_logger import log_model_results, get_git_commit
import json
from datetime import datetime

def evaluate_cnn():
    data_dir = 'data/raw'
    split_file = 'data/processed/split_assignment.json'
    
    if not os.path.exists(data_dir):
        print("Data not found. Run Phase 1 first.")
        return
        
    # We load the entire 'train' split dataset, then split into cv folds internally
    dataset = ApneaECGDataset(data_dir=data_dir, split='train', split_file=split_file)
    if len(dataset) == 0:
        print("No training data available.")
        return
        
    import json
    with open(split_file, 'r') as f:
        all_splits = json.load(f)
    train_records = [rec_id for rec_id, s in all_splits.items() if s == 'train']
    
    cv_splits = subject_level_kfold(train_records, k=5)
    
    manifest_path = 'models/artifacts/manifest.json'
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = []
        
    git_commit = get_git_commit()
    trained_at = datetime.now().isoformat()
    
    # Pre-calculate record_id for each index in the dataset to map folds correctly
    record_map = []
    for rec_id in dataset.record_ids:
        rec_path = os.path.join(data_dir, rec_id)
        if not os.path.exists(rec_path + '.dat'):
            continue
        import wfdb
        from src.data.annotations import parse_apnea_annotations
        record = wfdb.rdrecord(rec_path)
        fs = record.fs
        annotations = parse_apnea_annotations(rec_id, data_dir)
        if annotations.empty:
            continue
        samples_per_minute = fs * 60
        num_minutes = min(record.sig_len // samples_per_minute, len(annotations))
        record_map.extend([rec_id] * num_minutes)
        
    record_map = np.array(record_map)
    
    metrics = {'precision': [], 'recall': [], 'f1': [], 'pr_auc': []}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_ids, val_ids) in enumerate(cv_splits):
        print(f"Training Fold {fold+1}/5...")
        
        train_idx = np.where(np.isin(record_map, train_ids))[0]
        val_idx = np.where(np.isin(record_map, val_ids))[0]
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        model = Small1DCNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        epochs = 20
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x).squeeze(-1)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x).squeeze(-1)
                    val_loss += criterion(out, y).item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        # Load best weights
        model.load_state_dict(best_model_state)
        
        # Evaluate on val fold
        model.eval()
        y_true = []
        y_prob = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x).squeeze(-1)
                prob = torch.sigmoid(out).cpu().numpy()
                y_true.extend(y.numpy())
                y_prob.extend(prob)
                
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob > 0.5).astype(int)
        
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        metrics['pr_auc'].append(average_precision_score(y_true, y_prob))
        
    prec = (np.mean(metrics['precision']), np.std(metrics['precision']))
    rec = (np.mean(metrics['recall']), np.std(metrics['recall']))
    f1 = (np.mean(metrics['f1']), np.std(metrics['f1']))
    prauc = (np.mean(metrics['pr_auc']), np.std(metrics['pr_auc']))
    
    log_model_results("1D-CNN (Raw ECG)", prec, rec, f1, prauc)
    print("Logged 1D-CNN metrics.")
    
    # Retrain on full dataset
    print("Retraining 1D-CNN on full dataset...")
    full_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    final_model = Small1DCNN().to(device)
    criterion_final = nn.BCEWithLogitsLoss()
    optimizer_final = optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    final_model.train()
    for epoch in range(10): # Fixed epochs since we have no internal val set
        for x, y in full_loader:
            x, y = x.to(device), y.to(device)
            optimizer_final.zero_grad()
            out = final_model(x).squeeze(-1)
            loss = criterion_final(out, y)
            loss.backward()
            optimizer_final.step()
            
    # Save model
    filename = f"cnn_1d_{git_commit}.pt"
    filepath = os.path.join('models', 'artifacts', filename)
    torch.save(final_model.state_dict(), filepath)
    print(f"Saved CNN artifact to {filepath}")
    
    # Append to manifest
    manifest.append({
        "model_name": "1D-CNN (Raw ECG)",
        "filename": filename,
        "git_commit_hash": git_commit,
        "trained_at": trained_at,
        "metrics": {
            "pr_auc": float(np.mean(metrics['pr_auc'])),
            "f1": float(np.mean(metrics['f1']))
        },
        "is_production": False
    })
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == '__main__':
    evaluate_cnn()
