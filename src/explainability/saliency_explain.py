import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from captum.attr import IntegratedGradients
from dataclasses import dataclass

@dataclass
class SaliencyExplanation:
    importance_array: np.ndarray
    plot_path: str

def explain_cnn_prediction(model, raw_window: np.ndarray, output_dir: str, example_id: str) -> SaliencyExplanation:
    """
    Explain a single prediction from the 1D-CNN using Integrated Gradients.
    raw_window shape: (6000,) or (1, 6000)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    if raw_window.ndim == 1:
        x_tensor = torch.tensor(raw_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    elif raw_window.ndim == 2:
        x_tensor = torch.tensor(raw_window, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        x_tensor = torch.tensor(raw_window, dtype=torch.float32).to(device)
        
    x_tensor.requires_grad = True
    
    ig = IntegratedGradients(model)
    # We attribute with respect to target class 0 (since output is 1 neuron representing prob of class 1)
    attributions, delta = ig.attribute(x_tensor, target=None, return_convergence_delta=True)
    
    attr_np = attributions.squeeze().cpu().detach().numpy()
    sig_np = x_tensor.squeeze().cpu().detach().numpy()
    
    plot_path = os.path.join(output_dir, f'saliency_{example_id}.png')
    
    plt.figure(figsize=(10, 4))
    
    # We plot the raw signal, overlaid with absolute attribution intensity
    intensity = np.abs(attr_np)
    # Normalize intensity for visualization
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)
        
    time_axis = np.arange(len(sig_np))
    
    plt.plot(time_axis, sig_np, color='black', alpha=0.5, label='Filtered ECG')
    plt.scatter(time_axis, sig_np, c=intensity, cmap='Reds', s=10, alpha=intensity, label='IG Saliency')
    
    plt.title(f'Integrated Gradients Saliency - {example_id}')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.colorbar(label='Normalized Importance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return SaliencyExplanation(importance_array=attr_np, plot_path=plot_path)
