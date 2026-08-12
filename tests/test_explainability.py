import pytest
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from src.explainability.shap_explain import explain_classical_prediction
from src.explainability.saliency_explain import explain_cnn_prediction

def test_explain_classical(tmp_path):
    X = pd.DataFrame(np.random.rand(10, 5), columns=[f'f{i}' for i in range(5)])
    y = np.random.randint(0, 2, size=10)
    
    model = RandomForestClassifier(n_estimators=10, max_depth=2).fit(X, y)
    
    row = X.iloc[0]
    out_dir = str(tmp_path)
    
    res = explain_classical_prediction(model, row, out_dir, "test_classical")
    
    assert len(res.top_features) > 0
    assert os.path.exists(res.plot_path)
    assert res.plot_path.endswith('.png')

def test_explain_cnn(tmp_path):
    class DummyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 1)
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
            
    model = DummyCNN()
    window = np.random.rand(100).astype(np.float32)
    
    out_dir = str(tmp_path)
    res = explain_cnn_prediction(model, window, out_dir, "test_cnn")
    
    assert res.importance_array.shape == (100,)
    assert os.path.exists(res.plot_path)
