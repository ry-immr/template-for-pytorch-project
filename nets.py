import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.p = nn.Parameter(torch.zeros((1)))


    def forward(self, x):
        x = self.p * x
        return x
