import os
import sys

# Ensure src module is reachable when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    table_path = 'data/processed/feature_table.parquet'
    if not os.path.exists(table_path):
        print(f"{table_path} not found. Run build_feature_table.py first.")
        return
        
    df = pd.read_parquet(table_path)
    
    # NaN Rate Summary
    nan_rates = df.isna().mean() * 100
    features = [c for c in df.columns if c not in ['record_id', 'minute_index', 'split', 'label', 'is_spo2_available']]
    
    report = "# Feature Analysis Report\n\n"
    report += "## NaN Rates\n\n"
    report += "| Column | NaN Rate (%) | Explanation |\n"
    report += "|--------|--------------|-------------|\n"
    
    for col in df.columns:
        if col in ['record_id', 'minute_index', 'split', 'label']:
            continue
        rate = nan_rates[col]
        
        explanation = "N/A"
        if col == 'is_spo2_available':
            explanation = "Should be 0%."
        elif col.startswith('spo2_'):
            explanation = "NaNs occur when `is_spo2_available` is False, explicitly propagating the absence of the SpO2 channel rather than mixing them with genuine zero-dip windows."
        elif col in ['SDNN', 'RMSSD', 'pNN50', 'LF_HF_ratio']:
            explanation = "NaNs occur when a 1-minute window contains fewer than 3 R-peaks (e.g., severe artifact, disconnect, or extreme bradycardia), making HRV computation mathematically invalid."
            if col == 'LF_HF_ratio':
                explanation += " LF/HF can also be NaN if the frequency components are zero or cannot be resolved in a noisy window."
                
        report += f"| {col} | {rate:.2f}% | {explanation} |\n"
        
    docs_dir = 'docs'
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, 'nan_rate_summary.md')
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"Saved NaN rate summary to {report_path}")
    
    # Correlation Heatmap on TRAIN ONLY
    # This explicit slicing enforces rigorous leakage-avoidance discipline.
    train_df = df[df['split'] == 'train'].copy()
    train_df['label_bin'] = (train_df['label'] == 'apnea').astype(int)
    
    corr_cols = features + ['label_bin']
    corr_matrix = train_df[corr_cols].corr(method='spearman') # Spearman handles non-linear and outliers better
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap (Train Split Only)')
    plt.tight_layout()
    
    heatmap_path = os.path.join(docs_dir, 'feature_correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    
    print(f"Saved correlation heatmap to {heatmap_path}")

if __name__ == '__main__':
    main()
