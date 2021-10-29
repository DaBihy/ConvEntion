import torch.nn as nn
from .single import Attention
from einops import rearrange
import torch

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model[1] % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model[1]
        self.h = h
        self.configs = d_model

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.d_k, out_channels=3*self.d_k, kernel_size=1)
        )
        # self.output_linear = nn.Linear(self.d_k, self.d_k)
        self.attention = Attention(self.d_k, self.h)
        self.out_conv = nn.Conv2d(in_channels=self.d_k, out_channels=self.d_k, kernel_size=3,stride=1,padding=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, mask=None):
         # b c h w l   --> b 3*he*n h w l
        b,n,h,w,l = query.shape
        qkv_setlist = []
        
        # 1)Do all the linear projections in batch from d_model: b c h w l   --> b 3*he*n h w l
        for i in range(l):
            qkv_setlist.append(self.conv1(query[...,i]))
        qkv_set = torch.stack(qkv_setlist,dim=-1)
        query,key,value = torch.split(qkv_set,self.d_k ,dim=1)

        # 2) Apply attention on all the projected tensor in batch.
        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "TODO Concat" using a view and apply a final linear.
     
        # x = rearrange(x, 'b n h w l -> b l h w n')
        # x = rearrange(self.dropout(self.output_linear(x)), 'b l h w n -> b n h w l', b=b,n=n,h=h,w=w,l=l)
        
        out_setlist = []
        for i in range(l):
            out_setlist.append(self.out_conv(x[...,i]))
        x = torch.stack(out_setlist,dim=-1)

        return x
