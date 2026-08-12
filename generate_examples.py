import os
import sys
from sqlalchemy.orm import Session
from api.db import SessionLocal
from api.services.ml_pipeline import run_predictions, get_or_compute_explanation

def main():
    from api.services.model_loader import load_all_models
    load_all_models()
    
    db = SessionLocal()
    
    # We downloaded 'a01' during quick_train
    rec_id = "a01"
    
    print(f"Running CNN predictions for {rec_id}...")
    try:
        cnn_preds = run_predictions(rec_id, db, model_version="cnn")
        print(f"CNN predictions length: {len(cnn_preds)}")
        if cnn_preds:
            print(f"Generating Saliency for prediction {cnn_preds[0].id}...")
            exp = get_or_compute_explanation(cnn_preds[0].id, db)
            print(f"Saliency plot saved to: {exp.plot_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CNN Error: {e}")
        
    print(f"Running Classical predictions for {rec_id}...")
    try:
        rf_preds = run_predictions(rec_id, db, model_version="classical")
        print(f"RF predictions length: {len(rf_preds)}")
        if rf_preds:
            print(f"Generating SHAP for prediction {rf_preds[0].id}...")
            exp = get_or_compute_explanation(rf_preds[0].id, db)
            print(f"SHAP plot saved to: {exp.plot_path}")
            print(f"Top features: {exp.top_features}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Classical Error: {e}")

if __name__ == "__main__":
    main()
