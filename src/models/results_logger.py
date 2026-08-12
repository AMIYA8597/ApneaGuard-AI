import os
import subprocess
from datetime import datetime

def get_git_commit() -> str:
    """Get the short git commit hash, with a fallback that explicitly indicates local uncommitted state."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        return commit_hash
    except Exception:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"local-uncommitted-{timestamp}"

def log_model_results(model_name: str, precision: tuple, recall: tuple, f1: tuple, pr_auc: tuple):
    """
    Append run metrics to docs/model_results.md automatically.
    The input metrics should be tuples of (mean, std).
    """
    out_file = 'docs/model_results.md'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    commit_hash = get_git_commit()
        
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not os.path.exists(out_file):
        with open(out_file, 'w') as f:
            f.write("# Model Results\n\n")
            f.write("| Model | Commit | Date | Precision | Recall | F1 Score | PR-AUC |\n")
            f.write("|-------|--------|------|-----------|--------|----------|--------|\n")
            
    with open(out_file, 'a') as f:
        f.write(f"| {model_name} | {commit_hash} | {run_date} | "
                f"{precision[0]:.4f} ± {precision[1]:.4f} | "
                f"{recall[0]:.4f} ± {recall[1]:.4f} | "
                f"{f1[0]:.4f} ± {f1[1]:.4f} | "
                f"{pr_auc[0]:.4f} ± {pr_auc[1]:.4f} |\n")
