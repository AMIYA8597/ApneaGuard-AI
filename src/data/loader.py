import os
import glob
import wfdb

def download_apnea_ecg(dl_dir: str) -> list[str]:
    """
    Download the apnea-ecg database using wfdb and return the list of 
    record IDs that were actually downloaded.
    """
    os.makedirs(dl_dir, exist_ok=True)
    wfdb.dl_database('apnea-ecg', dl_dir=dl_dir)
    
    # Read what was actually downloaded
    hea_files = glob.glob(os.path.join(dl_dir, '*.hea'))
    record_ids = [os.path.splitext(os.path.basename(f))[0] for f in hea_files]
    return sorted(record_ids)
