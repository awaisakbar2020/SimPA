import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_dim1=128, hidden_dim2=64, output_size=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim1)  # Input: 3 features
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Hidden Layer 1: 128 units
        self.fc3 = nn.Linear(hidden_dim2, output_size)  # Hidden Layer 2: 64 units, Output: 4 actions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
