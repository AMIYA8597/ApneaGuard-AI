import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass

@dataclass
class ShapExplanation:
    top_features: list[tuple[str, float]]
    plot_path: str

def explain_classical_prediction(model, feature_row: pd.Series, output_dir: str, example_id: str) -> ShapExplanation:
    """
    Explain a single prediction from a tree-based classical ML model using SHAP.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    # feature_row must be a dataframe of 1 row for shap
    X = pd.DataFrame(feature_row).T
    
    shap_values = explainer.shap_values(X)
    
    # Depending on model (e.g. LightGBM), shap_values might be a list (per class) or an array
    if isinstance(shap_values, list):
        vals = shap_values[1][0] # Focus on positive class
    else:
        # If it's a single array (e.g., binary classification returning 1 output)
        if len(shap_values.shape) == 3: # (1, num_features, num_classes)
            vals = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = shap_values
            
    # Extract top features
    feature_names = X.columns
    importance = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)
    
    plot_path = os.path.join(output_dir, f'shap_{example_id}.png')
    
    # Save a force plot or waterfall
    plt.figure()
    # Ensure shap values match expected format for waterfall
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    if isinstance(base_val, (np.ndarray, list)):
        base_val = float(base_val[0])
        
    exp = shap.Explanation(values=vals, base_values=base_val, data=X.iloc[0].values, feature_names=feature_names.tolist())
    
    shap.waterfall_plot(exp, show=False)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return ShapExplanation(top_features=importance[:5], plot_path=plot_path)
