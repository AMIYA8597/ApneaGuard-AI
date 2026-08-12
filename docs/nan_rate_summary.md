# Feature Analysis Report

## NaN Rates

| Column | NaN Rate (%) | Explanation |
|--------|--------------|-------------|
| is_spo2_available | 0.00% | Should be 0%. |
| SDNN | 0.32% | NaNs occur when a 1-minute window contains fewer than 3 R-peaks (e.g., severe artifact, disconnect, or extreme bradycardia), making HRV computation mathematically invalid. |
| RMSSD | 0.32% | NaNs occur when a 1-minute window contains fewer than 3 R-peaks (e.g., severe artifact, disconnect, or extreme bradycardia), making HRV computation mathematically invalid. |
| pNN50 | 0.32% | NaNs occur when a 1-minute window contains fewer than 3 R-peaks (e.g., severe artifact, disconnect, or extreme bradycardia), making HRV computation mathematically invalid. |
| LF_HF_ratio | 100.00% | NaNs occur when a 1-minute window contains fewer than 3 R-peaks (e.g., severe artifact, disconnect, or extreme bradycardia), making HRV computation mathematically invalid. LF/HF can also be NaN if the frequency components are zero or cannot be resolved in a noisy window. |
| spo2_dip_count | 100.00% | NaNs occur when `is_spo2_available` is False, explicitly propagating the absence of the SpO2 channel rather than mixing them with genuine zero-dip windows. |
| spo2_mean_dip_depth | 100.00% | NaNs occur when `is_spo2_available` is False, explicitly propagating the absence of the SpO2 channel rather than mixing them with genuine zero-dip windows. |
| spo2_mean_dip_duration | 100.00% | NaNs occur when `is_spo2_available` is False, explicitly propagating the absence of the SpO2 channel rather than mixing them with genuine zero-dip windows. |
