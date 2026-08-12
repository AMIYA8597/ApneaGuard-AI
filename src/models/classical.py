import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from src.models.cross_validation import subject_level_kfold
from src.models.results_logger import log_model_results

def evaluate_classical_models():
    table_path = 'data/processed/feature_table.parquet'
    if not os.path.exists(table_path):
        print("Feature table not found. Run build_feature_table.py first.")
        return
        
    df = pd.read_parquet(table_path)
    train_df = df[df['split'] == 'train'].copy()
    
    if len(train_df) == 0:
        print("No training data available.")
        return
        
    train_df['label_bin'] = (train_df['label'] == 'apnea').astype(int)
    features = [c for c in train_df.columns if c not in ['record_id', 'minute_index', 'split', 'label', 'label_bin']]
    
    train_df[features] = train_df[features].fillna(train_df[features].mean())
    
    unique_records = train_df['record_id'].unique().tolist()
    cv_splits = subject_level_kfold(unique_records, k=5)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    # For tracking the best model's confusion matrix
    best_f1_score = -1.0
    best_model_name = ""
    best_cm_aggregate = np.zeros((2, 2))
    
    for model_name, model in models.items():
        metrics = {'precision': [], 'recall': [], 'f1': [], 'pr_auc': []}
        cm_aggregate = np.zeros((2, 2))
        
        for train_ids, val_ids in cv_splits:
            train_fold = train_df[train_df['record_id'].isin(train_ids)]
            val_fold = train_df[train_df['record_id'].isin(val_ids)]
            
            X_train, y_train = train_fold[features], train_fold['label_bin']
            X_val, y_val = val_fold[features], val_fold['label_bin']
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            metrics['pr_auc'].append(average_precision_score(y_val, y_prob))
            
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
            cm_aggregate += cm
            
        mean_f1 = np.mean(metrics['f1'])
        if mean_f1 > best_f1_score:
            best_f1_score = mean_f1
            best_model_name = model_name
            best_cm_aggregate = cm_aggregate
            
        prec = (np.mean(metrics['precision']), np.std(metrics['precision']))
        rec = (np.mean(metrics['recall']), np.std(metrics['recall']))
        f1 = (mean_f1, np.std(metrics['f1']))
        prauc = (np.mean(metrics['pr_auc']), np.std(metrics['pr_auc']))
        
        log_model_results(model_name, prec, rec, f1, prauc)
        print(f"Logged {model_name} metrics.")
        
    # Save the best model's confusion matrix
    docs_dir = 'docs'
    os.makedirs(docs_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm_aggregate, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'])
    plt.title(f'Confusion Matrix: {best_model_name} (Cross-Validated)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = os.path.join(docs_dir, 'confusion_matrix_classical.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved best model confusion matrix to {cm_path}")

if __name__ == '__main__':
    evaluate_classical_models()
