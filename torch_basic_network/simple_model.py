import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, hidden_size=250, input_size=15):
        super(SimpleModel, self).__init__()
        output_size = input_size * (input_size - 1)
        self.layers = nn.Sequential(
            nn.Linear(input_size*4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, m):
        return self.layers(m)