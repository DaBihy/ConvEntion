import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange

class Attention(nn.Module):
    """
    Compute ' Attention
    """
    def __init__(self,num_hidden, heads):
        super(Attention, self).__init__()

        self.num_hidden =num_hidden
        self.heads = heads

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_hidden, out_channels=self.heads, kernel_size=5,padding=2)
        )

    def forward(self, query, key, value, mask=None, dropout=None):

        b,n,h,w,l = query.shape
        c = n // self.heads
        val = value.view(b,self.heads, c, h, w, l)
        Vout_list = []
        for i in range(l):
            Qi = rearrange(torch.stack([query[...,i]]*l, dim=-1) + key, 'b n h w l -> (b l) n h w')
            tmp = rearrange(self.conv2(Qi),'(b l) n h w -> b n h w l',l=l)
            tmp = tmp.view(b,self.heads, 1, h, w, l)
            if mask is not None:
                tmp[~mask] = float('-inf')
            tmp = F.softmax(tmp, dim=5) #(b, heads, 1, h, w, l)
            tmp = tmp*val
            Vout_list.append(torch.sum(tmp, dim=5)) #(b, heads, c, h, w)
        Vout = torch.stack(Vout_list, dim=-1)#(b, heads, c, h, w, l)

        return Vout.view(b,self.heads*c, h, w, l)
