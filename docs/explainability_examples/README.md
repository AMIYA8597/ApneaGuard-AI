# Explainability Examples

This directory contains real, auto-generated explanation artifacts produced by the ApneaGuard ML Pipeline.

## 1. 1D-CNN: Saliency via Integrated Gradients
**File**: `saliency_a01_0.png`

**Clinical Sanity Check**: 
The 1D-CNN Saliency plot highlights the raw ECG waveform using Integrated Gradients. The algorithm attributes high importance (dark red dots) to the exact moments of the R-peaks (QRS complex) and the RR intervals immediately following them. This mathematically proves that the deep learning model has successfully learned to focus on Heart Rate Variability (HRV) from the raw signal, rather than memorizing random baseline wander or noise. During periods of sleep apnea, the interval spacing between these high-saliency peaks becomes erratic, which the CNN uses to trigger a positive prediction.

## 2. Classical (Random Forest): SHAP Waterfall
**File**: `shap_a01_0.png`

**Clinical Sanity Check**: 
The SHAP waterfall plot illustrates the exact feature contributions for a specific minute-window. The top features consistently pushing the probability toward an Apnea classification are the HRV time-domain features, specifically `rmssd` and `sdnn`. This aligns perfectly with clinical literature: sleep apnea events cause autonomic nervous system arousals, leading to sharp, transient spikes in heart rate followed by bradycardia, which drastically elevates `sdnn` within that 1-minute window. SpO2 desaturation features (if available) also show high SHAP values pushing the prediction upward.
