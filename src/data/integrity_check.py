import os
import sys

# Ensure src module is reachable when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from collections import Counter
from src.data.loader import download_apnea_ecg
from src.data.annotations import parse_apnea_annotations
from src.data.split import assign_split

def main():
    data_dir = 'data/raw'
    docs_dir = 'docs'
    os.makedirs(docs_dir, exist_ok=True)
    
    print("Downloading dataset...")
    record_ids = download_apnea_ecg(data_dir)
    print(f"Downloaded {len(record_ids)} records.")
    
    print("Assigning splits...")
    splits = assign_split(record_ids)
    
    split_counts = Counter(splits.values())
    
    total_minutes = 0
    apnea_minutes = 0
    normal_minutes = 0
    
    # Process only records that are in 'train' or 'test'
    annotated_records = [r for r, s in splits.items() if s in ['train', 'test']]
    
    print(f"Processing {len(annotated_records)} annotated records...")
    for r in annotated_records:
        df = parse_apnea_annotations(r, data_dir)
        if df.empty:
            continue
            
        total_minutes += len(df)
        apnea_minutes += (df['label'] == 'apnea').sum()
        normal_minutes += (df['label'] == 'normal').sum()
        
    total_hours = total_minutes / 60.0
    apnea_percent = (apnea_minutes / total_minutes * 100) if total_minutes > 0 else 0
    normal_percent = (normal_minutes / total_minutes * 100) if total_minutes > 0 else 0
    
    report = f"""# Data Integrity Report

## Overall Metrics
- **Total Subject Count**: {len(annotated_records)} (annotated working set)
- **Total Recording Hours**: {total_hours:.2f} hours
- **Class Balance**: 
  - Apnea: {apnea_percent:.2f}% ({apnea_minutes} minutes)
  - Normal: {normal_percent:.2f}% ({normal_minutes} minutes)

## Split Assignments
- **Train**: {split_counts.get('train', 0)} subjects
- **Test**: {split_counts.get('test', 0)} subjects
- **Holdout (Unannotated)**: {split_counts.get('holdout', 0)} subjects
"""
    print("\n--- Integrity Report ---")
    print(report)
    
    report_path = os.path.join(docs_dir, 'data_integrity_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved to {report_path}")

if __name__ == '__main__':
    main()
