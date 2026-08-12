# Model Artifacts

This directory stores the trained binary model artifacts (`.joblib` for classical ML models, `.pt` for PyTorch deep learning models). 

Because these files are large, they are excluded from Git via `.gitignore`. 

## How to Regenerate
To regenerate the exact models, run the following scripts from the project root. The models will be retrained, scored via subject-level cross-validation, and saved automatically with a Git commit provenance hash in their filename:

```bash
# Regenerate Classical Baselines (XGBoost, LightGBM, Random Forest)
python src/models/classical.py

# Regenerate Deep Learning Model (1D-CNN)
python src/models/train_cnn.py
```

The metadata for each generated model, including its performance metrics and `is_production` status, is tracked in `manifest.json`.
