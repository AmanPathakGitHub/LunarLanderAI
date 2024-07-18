import torch.nn as nn

import torch.nn.functional as F

class LanderModel(nn.Module):
    
    def __init__(self, n_hidden):
        super(LanderModel, self).__init__()
        n_inputs = 8
        n_outputs = 4
        
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out
