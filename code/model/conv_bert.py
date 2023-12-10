import torch.nn as nn
import torch
from .transformer import TransformerBlock
from .embedding.pos_encoding import PositionalEncoding
from .feature_generator import feature_generator
from einops import rearrange

class ConvBERT(nn.Module):

    def __init__(self, hidden, n_layers, attn_heads, dropout=0.1, cnn_stack=3):
        """
        :param hidden: hidden size of the SeqBERT model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden[1] * 2
        self.cnn_stack = cnn_stack

        # Layers 
        self.embedding_band = nn.Embedding(10,1024, padding_idx=0)
        self.embedding = PositionalEncoding(configs=hidden)
        self.feature_embedding = feature_generator(hidden)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden , dropout) for _ in range(n_layers)])

    def forward(self, x, mask, band):
        b,n,h,w,l = x.shape
        curr_band = self.embedding_band(band)
        curr_band = torch.squeeze(curr_band)
        curr_band = rearrange(torch.unsqueeze(curr_band,2),'b l n (h w)  -> b n h w l', b=b,n=1,h=h,w=w,l=l)
        x = torch.cat((x,curr_band),1) 

        gen_img = []
        for i in range(l//self.cnn_stack):
            gen_img.append(torch.squeeze(self.feature_embedding(x[...,i*self.cnn_stack:(i+1)*self.cnn_stack])))

        x = torch.stack(gen_img, dim=-1)
        x = self.embedding(x)
        
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
