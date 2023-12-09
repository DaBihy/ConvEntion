import torch
import torch.nn as nn
import torch.nn.functional as F

class EuclidLoss(nn.Module):
    
    def __init__(self):
        super(EuclidLoss, self).__init__()

    def forward(self, input, target):

        x = F.normalize(input, dim=1)
        y = F.normalize(target, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
       