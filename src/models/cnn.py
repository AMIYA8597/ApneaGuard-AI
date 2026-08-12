import torch
import torch.nn as nn
import torch.nn.functional as F

class Small1DCNN(nn.Module):
    """
    A small 1D Convolutional Neural Network for raw ECG classification.
    
    Architecture Note:
    This dataset (35 annotated records) is very small by deep-learning standards. 
    An oversized network (e.g., deep ResNet) risks severe overfitting far more than underfitting. 
    Therefore, a "small and regularized" architecture with just 3 convolutional blocks 
    and heavy dropout is a deliberate choice for generalization, not a limitation to apologize for.
    """
    def __init__(self, input_channels=1, input_length=6000): # fs=100 -> 6000 samples/min
        super(Small1DCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x
