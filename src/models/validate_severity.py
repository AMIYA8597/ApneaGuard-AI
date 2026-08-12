import os
import sys
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.severity import WindowPrediction, compute_severity

def main():
    table_path = 'data/processed/feature_table.parquet'
    if not os.path.exists(table_path):
        print("Feature table not found. Run build_feature_table.py first.")
        return
        
    df = pd.read_parquet(table_path)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Missing train or test data.")
        return
        
    train_df['label_bin'] = (train_df['label'] == 'apnea').astype(int)
    test_df['label_bin'] = (test_df['label'] == 'apnea').astype(int)
    
    features = [c for c in train_df.columns if c not in ['record_id', 'minute_index', 'split', 'label', 'label_bin']]
    
    # Impute NaNs with train mean
    mean_vals = train_df[features].mean()
    train_df[features] = train_df[features].fillna(mean_vals)
    test_df[features] = test_df[features].fillna(mean_vals)
    
    # Train Best Model (assuming LightGBM for speed and usual performance)
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(train_df[features], train_df['label_bin'])
    
    test_df['pred_prob'] = model.predict_proba(test_df[features])[:, 1]
    test_df['pred_class'] = (test_df['pred_prob'] > 0.5).astype(bool)
    
    # Evaluate per subject
    results = []
    
    for rec_id, group in test_df.groupby('record_id'):
        duration = len(group) # 1 row = 1 minute
        
        true_preds = [
            WindowPrediction(minute_index=row['minute_index'], is_apnea=(row['label']=='apnea'), probability=1.0)
            for _, row in group.iterrows()
        ]
        
        model_preds = [
            WindowPrediction(minute_index=row['minute_index'], is_apnea=row['pred_class'], probability=row['pred_prob'])
            for _, row in group.iterrows()
        ]
        
        true_sev = compute_severity(true_preds, duration)
        pred_sev = compute_severity(model_preds, duration)
        
        results.append({
            'record_id': rec_id,
            'duration_hours': true_sev.duration_hours,
            'true_ahi': true_sev.ahi,
            'true_band': true_sev.severity_band,
            'pred_ahi': pred_sev.ahi,
            'pred_band': pred_sev.severity_band
        })
        
    # Write to docs/severity_validation.md
    out_path = 'docs/severity_validation.md'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w') as f:
        f.write("# Severity Validation\n\n")
        f.write("This table compares the ground-truth severity band (derived from actual labels) against the predicted severity band (from LightGBM) for every test-split subject.\n\n")
        f.write("| Record | Duration (hrs) | True AHI | True Band | Predicted AHI | Predicted Band | Match? |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for r in results:
            match = "✅" if r['true_band'] == r['pred_band'] else "❌"
            if r['true_band'] != r['pred_band']:
                # Determine if it's off by one
                bands = ["normal", "mild", "moderate", "severe"]
                t_idx = bands.index(r['true_band'])
                p_idx = bands.index(r['pred_band'])
                if abs(t_idx - p_idx) == 1:
                    match += " (Off by 1)"
                else:
                    match += " (Wildly off)"
                    
            f.write(f"| {r['record_id']} | {r['duration_hours']:.2f} | {r['true_ahi']:.1f} | {r['true_band']} | {r['pred_ahi']:.1f} | {r['pred_band']} | {match} |\n")
            
if __name__ == '__main__':
    main()
