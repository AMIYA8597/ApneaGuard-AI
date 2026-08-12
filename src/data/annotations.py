import os
import logging
import pandas as pd
import wfdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_apnea_annotations(record_id: str, data_dir: str) -> pd.DataFrame:
    """
    Parse the .apn annotation file for a given record.
    Returns a per-minute DataFrame with columns [minute_index, label].
    Label is 'apnea' or 'normal'.
    """
    record_path = os.path.join(data_dir, record_id)
    apn_file = f"{record_path}.apn"
    
    if not os.path.exists(apn_file):
        logger.warning(f"Annotation file missing for record {record_id}")
        return pd.DataFrame(columns=['minute_index', 'label'])
        
    try:
        ann = wfdb.rdann(record_path, 'apn')
        
        # Apnea-ECG annotations: 'A' -> apnea, 'N' -> normal
        df = pd.DataFrame({
            'minute_index': range(len(ann.symbol)),
            'label': ['apnea' if sym == 'A' else 'normal' for sym in ann.symbol]
        })
        return df
    except Exception as e:
        logger.error(f"Malformed annotation data for record {record_id}: {e}")
        return pd.DataFrame(columns=['minute_index', 'label'])
