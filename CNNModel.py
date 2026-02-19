import torch
import torch.nn as nn
import torch.nn.functional as F


class DNA_CNN(nn.Module):
    def __init__(self):
        super(DNA_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(5, 64, kernel_size=12)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # x: (batch_size, seq_len, 5)
        x = x.permute(0, 2, 1)  # -> (batch_size, 5, seq_len)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.pool(x).squeeze(-1)  # -> (batch_size, 256)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x 
