import torch.nn as nn

from copy import deepcopy

def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])