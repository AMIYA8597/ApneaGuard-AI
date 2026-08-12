import os
import json
import logging
from src.data.split import assign_split
from src.data.annotations import parse_apnea_annotations

def test_no_subject_in_both_train_and_test(tmp_path):
    # Mock some record IDs matching the working set and some unannotated
    record_ids = [f'a{i:02d}' for i in range(1, 21)] + \
                 [f'b{i:02d}' for i in range(1, 6)] + \
                 [f'c{i:02d}' for i in range(1, 11)] + \
                 ['x01', 'x02']
                 
    split_file = tmp_path / "split_assignment.json"
    
    splits = assign_split(record_ids, split_file=str(split_file))
    
    train_records = {k for k, v in splits.items() if v == 'train'}
    test_records = {k for k, v in splits.items() if v == 'test'}
    holdout_records = {k for k, v in splits.items() if v == 'holdout'}
    
    # Assert zero overlap
    assert len(train_records.intersection(test_records)) == 0
    assert len(train_records.intersection(holdout_records)) == 0
    assert len(test_records.intersection(holdout_records)) == 0
    
    # Ensure file was created
    assert os.path.exists(split_file)
    with open(split_file, 'r') as f:
        saved_splits = json.load(f)
    assert saved_splits == splits

def test_annotation_parsing_handles_missing_records(caplog):
    # Use caplog to check for logged warning
    with caplog.at_level(logging.WARNING):
        df = parse_apnea_annotations('nonexistent_record', 'dummy_dir')
        
    assert df.empty
    assert "Annotation file missing for record nonexistent_record" in caplog.text
