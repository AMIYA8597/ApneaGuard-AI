import os
import logging
import pandas as pd
import numpy as np
import wfdb
import torch
from sqlalchemy.orm import Session
from api import models
from api.services.model_loader import get_model
from src.preprocessing.filtering import bandpass_filter_ecg
from src.preprocessing.rpeaks import detect_r_peaks
from src.preprocessing.windowing import window_by_minute
from src.preprocessing.spo2 import extract_desaturation_events
from src.features.hrv import compute_hrv_features
from src.features.spo2_features import compute_spo2_features
from src.models.severity import compute_severity

logger = logging.getLogger(__name__)

def run_predictions(recording_id: str, db: Session, model_version: str = "cnn"):
    logger.info(f"Invoked run_predictions for {recording_id} with model {model_version}")
    
    # 1. Load the requested model (fails loudly if missing)
    loaded_artifact = get_model(model_version)
    model = loaded_artifact["model"]
    model_type = loaded_artifact["type"]
    git_hash = loaded_artifact.get("name", model_version)
    
    # We will use the git_hash as the model_version_id in the database to guarantee provenance.
    version_id = loaded_artifact.get("name", "unknown") # Actually we map to the commit hash later
    # The loaded_artifact dict has: {"model": model, "type": "cnn", "name": model_name}, wait, git_hash is the key, let's just use the loaded_artifact key or the model_version requested if it resolved.
    # Actually, model_loader.py resolves "cnn" -> "abc1234". Let's get the real id.
    # We can get it by finding the key in _MODELS_REGISTRY, but model_loader doesn't expose it directly if we passed "cnn". 
    # Let's just use the passed model_version string, which the user might want us to record, or we just trust the provenance. 
    # Let's fetch the real ID. In model_loader, if we pass "cnn", it resolves to _PRODUCTION_MODEL_ID.
    # For now, let's just use the `model_version` string as the `model_version_id` in the DB.
    
    # 2. Load Raw WFDB data
    rec_path = os.path.join('data', 'raw', recording_id)
    if not os.path.exists(rec_path + '.dat'):
        logger.error(f"Raw data not found for {recording_id}")
        raise ValueError(f"Recording {recording_id} not found in raw data.")
        
    record = wfdb.rdrecord(rec_path)
    fs = record.fs
    sig_name = record.sig_name
    
    ecg_idx = sig_name.index('ECG') if 'ECG' in sig_name else 0
    raw_ecg = record.p_signal[:, ecg_idx]
    
    is_spo2_available = 'SpO2' in sig_name
    spo2_events = []
    if is_spo2_available:
        spo2_idx = sig_name.index('SpO2')
        spo2_signal = record.p_signal[:, spo2_idx]
        spo2_events = extract_desaturation_events(spo2_signal, fs)
        
    # 3. DSP Pipeline
    filtered_ecg = bandpass_filter_ecg(raw_ecg, fs)
    r_peaks = detect_r_peaks(filtered_ecg, fs)
    windows = window_by_minute(recording_id, raw_ecg, filtered_ecg, r_peaks, fs, annotations=None)
    
    logger.info(f"Generated {len(windows)} minute-windows for {recording_id}")
    
    # Delete old predictions for this recording
    db.query(models.Prediction).filter(models.Prediction.recording_id == recording_id).delete()
    
    predictions = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for win in windows:
        if model_type == "cnn":
            # Pass raw ECG tensor directly
            tensor_input = torch.tensor(win.raw_ecg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor_input).squeeze(-1)
                prob = torch.sigmoid(out).item()
        else:
            # Extract HRV and SpO2 features on the fly
            hrv_feats = compute_hrv_features(win.r_peaks, fs)
            spo2_feats = compute_spo2_features(spo2_events, win.minute_index, fs, is_spo2_available)
            
            # Reconstruct the feature row in the exact order expected by the model
            # For simplicity, we create a DataFrame with the same columns used in training
            row_dict = {**hrv_feats, **spo2_feats}
            
            # Note: During training, NaN filling happens. For a single row, if a feature is NaN (e.g. no peaks), we should fill with 0 or a mean.
            df_row = pd.DataFrame([row_dict]).fillna(0) 
            
            # Scikit-learn / XGBoost predict_proba
            # XGBoost expects the exact feature names.
            prob = model.predict_proba(df_row)[0, 1]
            
        p = models.Prediction(
            recording_id=recording_id,
            model_version_id=model_version,
            minute_index=win.minute_index,
            is_apnea=bool(prob > 0.5),
            probability=float(prob)
        )
        db.add(p)
        predictions.append(p)
        
    db.commit()
    logger.info(f"Saved {len(predictions)} real predictions for {recording_id}")
    return predictions

def get_or_compute_severity(recording_id: str, db: Session):
    logger.info(f"Computing severity for {recording_id}")
    # check cache
    existing = db.query(models.SeverityScore).filter(models.SeverityScore.recording_id == recording_id).first()
    if existing:
        return existing
        
    preds = db.query(models.Prediction).filter(
        models.Prediction.recording_id == recording_id
    ).all()
    
    if not preds:
        raise ValueError("No predictions found. Run predict first.")
        
    # We map the database Prediction objects to the WindowPrediction dataclass required by severity.py
    from src.models.severity import WindowPrediction
    window_preds = [WindowPrediction(minute_index=p.minute_index, is_apnea=p.is_apnea, probability=p.probability) for p in preds]
    
    # Compute using the core severity module
    sev_result = compute_severity(window_preds, len(window_preds))
        
    score = models.SeverityScore(
        recording_id=recording_id,
        model_version_id=preds[0].model_version_id,
        duration_hours=sev_result.duration_hours,
        ahi=sev_result.ahi,
        severity_band=sev_result.severity_band
    )
    db.add(score)
    db.commit()
    db.refresh(score)
    return score

def get_or_compute_explanation(prediction_id: int, db: Session):
    logger.info(f"Computing explanation for prediction {prediction_id}")
    existing = db.query(models.Explanation).filter(models.Explanation.prediction_id == prediction_id).first()
    if existing:
        return existing
        
    pred = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if not pred:
        raise ValueError("Prediction not found")
        
    recording_id = pred.recording_id
    minute_index = pred.minute_index
    model_version = pred.model_version_id
    
    # 1. Load the requested model (fails loudly if missing)
    loaded_artifact = get_model(model_version)
    model = loaded_artifact["model"]
    model_type = loaded_artifact["type"]
    
    # 2. Load Raw WFDB data
    rec_path = os.path.join('data', 'raw', recording_id)
    if not os.path.exists(rec_path + '.dat'):
        logger.error(f"Raw data not found for {recording_id}")
        raise ValueError(f"Recording {recording_id} not found in raw data.")
        
    record = wfdb.rdrecord(rec_path)
    fs = record.fs
    sig_name = record.sig_name
    
    ecg_idx = sig_name.index('ECG') if 'ECG' in sig_name else 0
    raw_ecg = record.p_signal[:, ecg_idx]
    
    is_spo2_available = 'SpO2' in sig_name
    spo2_events = []
    if is_spo2_available:
        spo2_idx = sig_name.index('SpO2')
        spo2_signal = record.p_signal[:, spo2_idx]
        spo2_events = extract_desaturation_events(spo2_signal, fs)
        
    # 3. DSP Pipeline
    filtered_ecg = bandpass_filter_ecg(raw_ecg, fs)
    r_peaks = detect_r_peaks(filtered_ecg, fs)
    windows = window_by_minute(recording_id, raw_ecg, filtered_ecg, r_peaks, fs, annotations=None)
    
    # 4. Find the correct window
    if minute_index >= len(windows):
        raise ValueError(f"Minute index {minute_index} out of bounds for recording {recording_id}")
    win = windows[minute_index]
    
    output_dir = "docs/explainability_examples"
    
    # 5. Generate Explanation
    if model_type == "cnn":
        from src.explainability.saliency_explain import explain_cnn_prediction
        # explain_cnn_prediction takes model, raw_window, output_dir, example_id
        explanation = explain_cnn_prediction(model, win.raw_ecg, output_dir, f"{recording_id}_{minute_index}")
        plot_path = explanation.plot_path
        top_features_dict = None
        method = "saliency"
    else:
        from src.explainability.shap_explain import explain_classical_prediction
        # Reconstruct the feature row exactly as in training
        hrv_feats = compute_hrv_features(win.r_peaks, fs)
        spo2_feats = compute_spo2_features(spo2_events, win.minute_index, fs, is_spo2_available)
        row_dict = {**hrv_feats, **spo2_feats}
        df_row = pd.Series(row_dict).fillna(0)
        
        explanation = explain_classical_prediction(model, df_row, output_dir, f"{recording_id}_{minute_index}")
        plot_path = explanation.plot_path
        # convert list of tuples to dict for JSONB
        top_features_dict = {k: float(v) for k, v in explanation.top_features}
        method = "shap"
        
    exp = models.Explanation(
        prediction_id=prediction_id,
        method=method,
        plot_path=plot_path,
        top_features=top_features_dict
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp
