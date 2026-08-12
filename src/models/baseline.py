import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from src.models.cross_validation import subject_level_kfold
from src.models.results_logger import log_model_results

def evaluate_baselines():
    table_path = 'data/processed/feature_table.parquet'
    if not os.path.exists(table_path):
        print("Feature table not found. Run build_feature_table.py first.")
        return
        
    df = pd.read_parquet(table_path)
    train_df = df[df['split'] == 'train'].copy()
    
    if len(train_df) == 0:
        print("No training data available.")
        return
        
    # Prepare features and labels
    train_df['label_bin'] = (train_df['label'] == 'apnea').astype(int)
    features = [c for c in train_df.columns if c not in ['record_id', 'minute_index', 'split', 'label', 'label_bin']]
    
    # Impute NaNs with mean (simple imputation for baseline)
    train_df[features] = train_df[features].fillna(train_df[features].mean())
    
    unique_records = train_df['record_id'].unique().tolist()
    cv_splits = subject_level_kfold(unique_records, k=5)
    
    majority_metrics = {'precision': [], 'recall': [], 'f1': [], 'pr_auc': []}
    logreg_metrics = {'precision': [], 'recall': [], 'f1': [], 'pr_auc': []}
    
    for train_ids, val_ids in cv_splits:
        train_fold = train_df[train_df['record_id'].isin(train_ids)]
        val_fold = train_df[train_df['record_id'].isin(val_ids)]
        
        X_train, y_train = train_fold[features], train_fold['label_bin']
        X_val, y_val = val_fold[features], val_fold['label_bin']
        
        # Majority Class Baseline
        majority_class = y_train.mode()[0]
        y_pred_maj = np.full_like(y_val, majority_class)
        # Probabilities: 1.0 if majority class is 1, else 0.0
        y_prob_maj = np.full_like(y_val, float(majority_class), dtype=float)
        
        # Accuracy alone would be misleading due to class imbalance documented in Phase 1.
        # We must track precision, recall, f1, and PR-AUC.
        majority_metrics['precision'].append(precision_score(y_val, y_pred_maj, zero_division=0))
        majority_metrics['recall'].append(recall_score(y_val, y_pred_maj, zero_division=0))
        majority_metrics['f1'].append(f1_score(y_val, y_pred_maj, zero_division=0))
        # Note: if y_prob is all exactly 0 or 1, PR-AUC might just be the baseline prevalence
        majority_metrics['pr_auc'].append(average_precision_score(y_val, y_prob_maj))
        
        # Logistic Regression Baseline
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_val)
        y_prob_lr = lr.predict_proba(X_val)[:, 1]
        
        logreg_metrics['precision'].append(precision_score(y_val, y_pred_lr, zero_division=0))
        logreg_metrics['recall'].append(recall_score(y_val, y_pred_lr, zero_division=0))
        logreg_metrics['f1'].append(f1_score(y_val, y_pred_lr, zero_division=0))
        logreg_metrics['pr_auc'].append(average_precision_score(y_val, y_prob_lr))
        
    for model_name, metrics in [("Majority Baseline", majority_metrics), ("Logistic Regression", logreg_metrics)]:
        prec = (np.mean(metrics['precision']), np.std(metrics['precision']))
        rec = (np.mean(metrics['recall']), np.std(metrics['recall']))
        f1 = (np.mean(metrics['f1']), np.std(metrics['f1']))
        prauc = (np.mean(metrics['pr_auc']), np.std(metrics['pr_auc']))
        log_model_results(model_name, prec, rec, f1, prauc)
        print(f"Logged {model_name} metrics.")

if __name__ == '__main__':
    evaluate_baselines()
