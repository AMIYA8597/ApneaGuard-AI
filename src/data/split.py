import os
import json
import random

def assign_split(record_ids: list[str], split_file: str = 'data/processed/split_assignment.json') -> dict[str, str]:
    """
    Assign record IDs to 'train', 'test', or 'holdout'.
    Writes assignment to a JSON file if not already present.
    """
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            return json.load(f)
            
    # The 35 records with released per-minute annotations: a01-a20, b01-b05, c01-c10
    annotated_prefixes = [f'a{i:02d}' for i in range(1, 21)] + \
                         [f'b{i:02d}' for i in range(1, 6)] + \
                         [f'c{i:02d}' for i in range(1, 11)]
                         
    annotated_records = [r for r in record_ids if r in annotated_prefixes]
    unannotated_records = [r for r in record_ids if r not in annotated_prefixes]
    
    random.seed(42)
    random.shuffle(annotated_records)
    
    # Hold out a subset of whole records as internal test split (e.g., 20%)
    n_test = int(0.2 * len(annotated_records))
    
    test_records = annotated_records[:n_test]
    train_records = annotated_records[n_test:]
    
    split_assignment = {}
    for r in train_records:
        split_assignment[r] = 'train'
    for r in test_records:
        split_assignment[r] = 'test'
    for r in unannotated_records:
        split_assignment[r] = 'holdout'
        
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    with open(split_file, 'w') as f:
        json.dump(split_assignment, f, indent=4)
        
    return split_assignment
