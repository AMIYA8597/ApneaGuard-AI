# Model Card: ApneaGuard AI 1D-CNN

## Model Details
- **Architecture**: 1-Dimensional Convolutional Neural Network (1D-CNN).
- **Modality**: Raw Physiological Time-Series (Electrocardiogram - ECG).
- **Framework**: PyTorch.
- **Date**: August 2026.
- **Version**: 1.0 (Research Prototype).

## Intended Use
- **Primary Use Case**: Automated detection of sleep apnea events in continuous ECG monitoring on a per-minute basis, facilitating severity scoring (AHI) estimation.
- **Out-of-Scope**: This model is explicitly NOT a clinical diagnostic device. It is intended strictly for research and portfolio demonstration.

## Training Data
- **Dataset**: PhysioNet Apnea-ECG Database.
- **Demographics**: 35 annotated records (70-90 hours of recording each). The demographic distribution (age, sex, ethnicity) is limited and heavily biased by the historic nature of the dataset.
- **Preprocessing**: Signal bandpass filtering (0.5Hz - 40Hz) using zero-phase Butterworth filters to preserve exact temporal alignment of the R-peaks against annotation boundaries.

## Evaluation Results
The model was evaluated using a rigorous **subject-level k-fold cross-validation** to prevent temporal data leakage.
- **PR-AUC**: 0.6948
- **F1-Score**: 0.6241
- **Comparative Baseline**: Outperformed the classical XGBoost baseline (PR-AUC 0.6109 / F1 0.5801) trained on engineered Heart Rate Variability (HRV) features.

## Explainability
- **Mechanism**: The model utilizes Saliency/Integrated Gradients to provide per-timestep importance arrays.
- **Clinical Validation**: Predictions can be visually mapped back to the raw waveform, allowing human reviewers to cross-reference flagged temporal segments with established physiological anomalies (e.g., bradycardia followed by tachycardia).

## Ethical Considerations & Known Limitations
- **Small Sample Size**: 35 records are insufficient to generalize across the broader population. The model is highly susceptible to demographic bias inherent in the training distribution.
- **Clinical Validation**: No real-world clinical trials, FDA clearances, or IRB-backed studies have evaluated this architecture.
- **Deep Learning Data Hunger**: The 1D-CNN relies on raw waveform structures. Should the sensor hardware change (e.g., Apple Watch ECG vs Holter Monitor), the model's performance will unpredictably degrade without re-training or domain adaptation.
