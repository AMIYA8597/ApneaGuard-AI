import os
import json
import joblib
import torch
import logging
from src.models.cnn import Small1DCNN

logger = logging.getLogger(__name__)

# Global registry of loaded models
_MODELS_REGISTRY = {}
_PRODUCTION_MODEL_ID = None

def load_all_models():
    """
    Called at API startup to preload all artifacts into memory.
    Fails LOUDLY if any expected file is missing to prevent silent fallback to heuristics.
    """
    global _MODELS_REGISTRY
    global _PRODUCTION_MODEL_ID
    
    from api.config import settings
    
    manifest_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found at {manifest_path}. Have models been trained?")
        raise FileNotFoundError(f"Missing model manifest: {manifest_path}")
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    for entry in manifest:
        model_name = entry["model_name"]
        filename = entry["filename"]
        git_hash = entry["git_commit_hash"]
        filepath = os.path.join(settings.MODEL_ARTIFACTS_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"CRITICAL: Artifact missing from disk: {filepath}")
            raise FileNotFoundError(f"Missing model artifact required by manifest: {filepath}")
            
        logger.info(f"Loading {model_name} (Commit: {git_hash}) from {filepath}")
        
        if filename.endswith(".pt"):
            # Load PyTorch model
            model = Small1DCNN().to(device)
            model.load_state_dict(torch.load(filepath, map_location=device))
            model.eval()
            _MODELS_REGISTRY[git_hash] = {"model": model, "type": "cnn", "name": model_name}
        elif filename.endswith(".joblib"):
            # Load Classical model
            model = joblib.load(filepath)
            _MODELS_REGISTRY[git_hash] = {"model": model, "type": "classical", "name": model_name}
        else:
            raise ValueError(f"Unknown artifact format for {filename}")
            
        if entry.get("is_production"):
            _PRODUCTION_MODEL_ID = git_hash
            logger.info(f"Set {model_name} ({git_hash}) as the active production model.")
            
    if not _MODELS_REGISTRY:
        raise RuntimeError("No models were loaded from the manifest!")
        
def get_model(model_version_id: str):
    """
    Retrieve a pre-loaded model by its git commit hash.
    If model_version_id is "default" or "production", returns the flagged model.
    """
    if model_version_id in ("default", "production", "cnn", "classical"):
        # For legacy compatibility, if someone requests "cnn" or "classical", 
        # just return the production one or a matching one
        if _PRODUCTION_MODEL_ID:
             model_version_id = _PRODUCTION_MODEL_ID
        else:
             model_version_id = list(_MODELS_REGISTRY.keys())[0]
             
    if model_version_id not in _MODELS_REGISTRY:
        raise KeyError(f"Requested model version {model_version_id} is not loaded.")
        
    return _MODELS_REGISTRY[model_version_id]
