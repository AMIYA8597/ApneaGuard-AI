import pytest
from src.models.severity import WindowPrediction, compute_severity

def test_compute_severity():
    # 480 minutes = 8 hours
    duration_minutes = 480 
    
    # Let's mock a scenario with 160 events over 8 hours (AHI = 20)
    # AHI = 20 -> moderate
    predictions = [
        WindowPrediction(minute_index=i, is_apnea=True, probability=0.9) 
        for i in range(160)
    ]
    
    res = compute_severity(predictions, duration_minutes)
    
    assert res.total_events == 160
    assert res.duration_hours == 8.0
    assert res.ahi == 20.0
    assert res.severity_band == "moderate"
    
    # Test Normal (AHI < 5)
    # 24 events / 8 hours = 3 AHI
    pred_normal = [WindowPrediction(minute_index=i, is_apnea=True, probability=0.9) for i in range(24)]
    res_normal = compute_severity(pred_normal, duration_minutes)
    assert res_normal.severity_band == "normal"
    
    # Test Mild (5 <= AHI < 15)
    # 80 events / 8 hours = 10 AHI
    pred_mild = [WindowPrediction(minute_index=i, is_apnea=True, probability=0.9) for i in range(80)]
    res_mild = compute_severity(pred_mild, duration_minutes)
    assert res_mild.severity_band == "mild"
    
    # Test Severe (AHI >= 30)
    # 320 events / 8 hours = 40 AHI
    pred_severe = [WindowPrediction(minute_index=i, is_apnea=True, probability=0.9) for i in range(320)]
    res_severe = compute_severity(pred_severe, duration_minutes)
    assert res_severe.severity_band == "severe"
