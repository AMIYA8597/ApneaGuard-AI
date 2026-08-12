import pytest
import numpy as np
import pandas as pd
import json
import os
from src.features.hrv import compute_hrv_features

def test_hrv_few_peaks_returns_nan(caplog):
    # Test that too few R-peaks return NaN safely
    r_peaks = np.array([100, 200]) # Only 2 peaks
    fs = 100
    
    result = compute_hrv_features(r_peaks, fs)
    
    assert np.isnan(result['SDNN'])
    assert np.isnan(result['RMSSD'])
    assert np.isnan(result['pNN50'])
    assert np.isnan(result['LF_HF_ratio'])
    
    assert "Too few R-peaks" in caplog.text

def test_feature_table_split_matches_assignment():
    # If the feature table exists, we verify that the 'split' column
    # strictly matches the assignment json.
    # This is a regression test against split-assignment drift.
    table_path = 'data/processed/feature_table.parquet'
    split_path = 'data/processed/split_assignment.json'
    
    if not os.path.exists(table_path) or not os.path.exists(split_path):
        pytest.skip("Data not yet generated")
        
    df = pd.read_parquet(table_path)
    
    with open(split_path, 'r') as f:
        splits = json.load(f)
        
    # Check that for every row, the split matches the assignment
    for idx, row in df.iterrows():
        rec_id = row['record_id']
        expected_split = splits.get(rec_id)
        assert row['split'] == expected_split, f"Split drift detected for record {rec_id}!"
