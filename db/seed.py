import os
import sys
import datetime
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.db import engine, SessionLocal
from api.models import Recording, ModelVersion, Prediction, SeverityScore, Explanation

def seed():
    print("Starting database seed...")
    with SessionLocal() as db:
        # Clear existing data (in Postgres, could use truncate, here we just delete)
        db.query(Explanation).delete()
        db.query(SeverityScore).delete()
        db.query(Prediction).delete()
        db.query(Recording).delete()
        db.query(ModelVersion).delete()
        db.commit()

        # Seed ModelVersions
        mv_class = ModelVersion(id="classical", git_commit_hash="abcdef12345", description="Baseline XGBoost model")
        mv_cnn = ModelVersion(id="cnn", git_commit_hash="fedcba54321", description="Deep 1D-CNN model")
        db.add_all([mv_class, mv_cnn])
        
        # Load a few records from parquet if available
        parquet_path = 'data/processed/feature_table.parquet'
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            # Take two records
            records_to_seed = df['record_id'].unique()[:2]
            
            for rec_id in records_to_seed:
                rec_df = df[df['record_id'] == rec_id]
                split_role = rec_df['split'].iloc[0] if 'split' in rec_df.columns else "train"
                duration = len(rec_df)
                
                rec = Recording(
                    id=rec_id,
                    status="completed",
                    duration_minutes=duration,
                    split_role=split_role
                )
                db.add(rec)
                
                preds = []
                for _, row in rec_df.iterrows():
                    is_apnea = bool(row['label'] == 'apnea') if 'label' in row.index else False
                    prob = 0.9 if is_apnea else 0.1
                    p = Prediction(
                        recording_id=rec_id,
                        model_version_id="cnn",
                        minute_index=int(row['minute_index']),
                        is_apnea=is_apnea,
                        probability=prob
                    )
                    preds.append(p)
                db.add_all(preds)
                
                # Compute mock severity
                ahi = sum(1 for p in preds if p.is_apnea) / (duration / 60.0) if duration > 0 else 0
                band = "normal" if ahi < 5 else "mild" if ahi < 15 else "moderate" if ahi < 30 else "severe"
                
                sev = SeverityScore(
                    recording_id=rec_id,
                    model_version_id="cnn",
                    duration_hours=duration / 60.0,
                    ahi=ahi,
                    severity_band=band
                )
                db.add(sev)
                
            db.commit()
            print(f"Seeded {len(records_to_seed)} records successfully.")
        else:
            print("Feature table not found. Seeded ModelVersions only.")

if __name__ == "__main__":
    seed()
