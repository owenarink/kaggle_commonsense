from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict
from src.modules import LinearModule


class CNN_TFIDF(nn.Module):
    def __init__(self, n_inputs: int, n_classes: int = 3):
        super().__init__()
        # input: [B, n_inputs] -> [B, 1, n_inputs]
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  
        return self.net(x)


      
