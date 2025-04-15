# whitebit/core/whitebit_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class WhiteBitNet(nn.Module):
    def __init__(self, precision_policy=None):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 256),  # input layer (28x28 flattened)
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 10)    # output layer
        ])
        self.precision_policy = precision_policy or [False, False, False, False]  # False = FP16, True = FP8

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input

        for i, layer in enumerate(self.layers[:-1]):
            precision = torch.float8_e4m3fn if self.precision_policy[i] else torch.float16
            x = layer(x.to(precision)).to(torch.float16)  # back to float16 for ReLU
            x = F.relu(x)

        # Output layer
        precision = torch.float8_e4m3fn if self.precision_policy[-1] else torch.float16
        x = self.layers[-1](x.to(precision))

        return x
