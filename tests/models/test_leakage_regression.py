import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.models.cross_validation import subject_level_kfold

def test_leakage_regression():
    """
    This test exists to prove the subject-level split is not just implemented 
    but load-bearing - that it actually changes the measured result relative to 
    the common (buggy) alternative. 
    
    If this test's premise stops holding (e.g., because the dataset changes), 
    investigate before assuming the split logic is still doing its job. We 
    simulate patient-specific leakage by giving each patient a unique 'quirk' 
    (threshold) that doesn't generalize to new patients.
    """
    np.random.seed(42)
    n_subjects = 20
    rows_per_subject = 100
    
    data = []
    for subj in range(n_subjects):
        record_id = f"subj_{subj}"
        
        # A proxy feature that identifies the patient (like baseline resting HR)
        patient_identifier = subj * 10
        
        # Even subjects have one rule, Odd subjects have the exact opposite rule.
        # This makes the "global" rule completely useless (50% accuracy).
        is_even = (subj % 2 == 0)
        
        for _ in range(rows_per_subject):
            hr_val = np.random.uniform(-2, 2)
            
            if is_even:
                label = 1 if hr_val > 0 else 0
            else:
                label = 1 if hr_val < 0 else 0
                
            data.append({
                "record_id": record_id,
                "hr_val": hr_val,
                "patient_identifier": patient_identifier,
                "label": label
            })
            
    df = pd.DataFrame(data)
    
    # Model capable of memorizing patient-specific features
    clf = DecisionTreeClassifier(random_state=42)
    
    # --- SCENARIO 1: The Buggy Random Row Split ---
    X = df[["hr_val", "patient_identifier"]]
    y = df["label"]
    
    # Randomly splitting rows leaks patient identities across train and test
    X_train_bug, X_test_bug, y_train_bug, y_test_bug = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    clf.fit(X_train_bug, y_train_bug)
    preds_bug = clf.predict(X_test_bug)
    acc_buggy = accuracy_score(y_test_bug, preds_bug)
    
    # --- SCENARIO 2: The Correct Subject-Level Split ---
    record_ids = df["record_id"].unique().tolist()
    # Ensuring subjects in test set never appear in train set
    folds = subject_level_kfold(record_ids, k=5)
    
    train_ids, test_ids = folds[0]
    
    train_df = df[df["record_id"].isin(train_ids)]
    test_df = df[df["record_id"].isin(test_ids)]
    
    X_train_correct = train_df[["hr_val", "patient_identifier"]]
    y_train_correct = train_df["label"]
    X_test_correct = test_df[["hr_val", "patient_identifier"]]
    y_test_correct = test_df["label"]
    
    clf.fit(X_train_correct, y_train_correct)
    preds_correct = clf.predict(X_test_correct)
    acc_correct = accuracy_score(y_test_correct, preds_correct)
    
    # --- ASSERTIONS ---
    # Buggy split should artificially look great because it memorized patient quirks
    assert acc_buggy > 0.90, f"Buggy split accuracy too low: {acc_buggy}"
    
    # Correct split should perform worse because thresholds don't generalize to new patients
    assert acc_correct < 0.70, f"Correct split accuracy suspiciously high: {acc_correct}"
    
    # The gap should be massive, proving the structural necessity of the split
    assert (acc_buggy - acc_correct) > 0.20, f"Gap not large enough! Buggy: {acc_buggy}, Correct: {acc_correct}"
