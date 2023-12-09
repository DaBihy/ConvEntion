import torch.nn as nn
# from .layer_norm import LayerNorm
from einops import rearrange


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm =  nn.LayerNorm(size[1:])

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."

        x1 = self.norm(x)
        x1 = sublayer(x1)
      
        return x + x1