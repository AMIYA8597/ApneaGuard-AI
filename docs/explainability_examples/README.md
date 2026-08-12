# Explainability Clinical Sanity Checks

These explanation artifacts represent 5 randomly sampled true-positive flagged apnea windows, analyzed using SHAP (for classical ML) and Integrated Gradients Saliency (for the 1D-CNN).

## 1. Classical ML (XGBoost) - Window A
**File**: `shap_classical_window_a.png`
**Clinical Sanity Check**: The SHAP waterfall plot shows that a severely depressed pNN50 feature and elevated LF/HF ratio dominated the prediction, which perfectly aligns with the sympathetic nervous system surge physiologically expected during an apnoeic event.

## 2. Classical ML (XGBoost) - Window B
**File**: `shap_classical_window_b.png`
**Clinical Sanity Check**: In this window, the model strongly relied on the SpO2 'mean dip duration' feature (which was unusually long). This is a credible and highly specific marker for severe obstructive events.

## 3. Deep Learning (1D-CNN) - Window C
**File**: `saliency_cnn_window_c.png`
**Clinical Sanity Check**: The saliency map highlights intense activation localized exactly over a prolonged period of bradycardia (extended R-R intervals) in the first 30 seconds of the window. This demonstrates the CNN has effectively learned to isolate heart rate variability anomalies directly from the raw waveform without manual feature engineering.

## 4. Deep Learning (1D-CNN) - Window D
**File**: `saliency_cnn_window_d.png`
**Clinical Sanity Check**: **(Mixed Result)** While the model correctly flagged this window as apnea, the saliency map reveals intense, spurious attention on a high-frequency baseline wander artifact near the end of the window, rather than the QRS complexes. This is a concerning sign that the CNN is occasionally overfitting to recording noise rather than true physiological markers.

## 5. Deep Learning (1D-CNN) - Window E
**File**: `saliency_cnn_window_e.png`
**Clinical Sanity Check**: The integrated gradients trace highlights both the initial bradycardia onset and the subsequent compensatory tachycardia burst perfectly, showing the model has learned the biphasic cardiac response to sleep-disordered breathing.
