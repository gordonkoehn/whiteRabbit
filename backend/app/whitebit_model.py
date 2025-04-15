# whitebit/core/whitebit_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class WhiteBitNet(nn.Module):
    def __init__(self, precision_policy=None):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 10)
        ])
        self.precision_policy = precision_policy or [False] * len(self.layers)  # False = FP16, True = Simulated FP8
        self.to(torch.float16)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input

        for i, layer in enumerate(self.layers[:-1]):
            if self.precision_policy[i]:
                print(f"[SIMULATED FP8] Using low-precision at layer {i}")

            # Force both data and layer to FP16 before forward
            #layer = layer.to(torch.float16)
            x = layer(x.to(torch.float16))  # Ensure input is in FP16
            x = F.relu(x)

        # Output layer
        if self.precision_policy[-1]:
            print(f"[SIMULATED FP8] Using low-precision at output layer")
        x = self.layers[-1](x.to(torch.float16))

        return x
