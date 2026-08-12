import pytest
from src.models.cross_validation import subject_level_kfold

def test_subject_level_kfold_no_overlap():
    record_ids = [f'rec_{i}' for i in range(100)]
    k = 5
    folds = subject_level_kfold(record_ids, k=k)
    
    assert len(folds) == k
    
    for train_ids, val_ids in folds:
        # 1. No overlap between train and val
        overlap = set(train_ids).intersection(set(val_ids))
        assert len(overlap) == 0, f"Leakage detected! Records in both splits: {overlap}"
        
        # 2. All records are accounted for in each fold
        assert set(train_ids).union(set(val_ids)) == set(record_ids)
