import numpy as np

def subject_level_kfold(record_ids: list[str], k: int = 5) -> list[tuple[list[str], list[str]]]:
    """
    Split record IDs into k folds. 
    
    Folding by window instead of by record is the most common
    bug in public sleep-apnea-detection code and silently inflates every
    downstream metric. This function exists specifically to make that bug
    structurally impossible.
    
    Returns a list of tuples: (train_record_ids, val_record_ids)
    """
    unique_records = list(set(record_ids))
    np.random.seed(42)
    np.random.shuffle(unique_records)
    
    folds = np.array_split(unique_records, k)
    
    cv_splits = []
    for i in range(k):
        val_records = folds[i].tolist()
        train_records = [r for r in unique_records if r not in val_records]
        cv_splits.append((train_records, val_records))
        
    return cv_splits
