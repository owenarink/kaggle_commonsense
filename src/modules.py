"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np
import torch.nn as nn
import torch.nn.init as init

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
  
    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
    
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity='relu') 
        init.zeros_(self.fc.bias)

    def forward(self, x):
      return self.fc(x) 
