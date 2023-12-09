
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class PositionalEncoding(nn.Module):

    def __init__(self, configs):
        super(PositionalEncoding, self).__init__()
        self.configs = configs
        self.num_hidden = configs[1]
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table())

    def _get_sinusoid_encoding_table(self):
        ''' Sinusoid position encoding table '''
        # 
        def get_position_angle_vec(position):

            return_list = [torch.ones((self.configs[2],
                                       self.configs[3]))*(position / np.power(10000, 2 * (hid_j // 2) / self.num_hidden)) for hid_j in range(self.num_hidden)]
            return torch.stack(return_list, dim=0)
        sinusoid_table = [get_position_angle_vec(pos_i) for pos_i in range(self.configs[4])]
        #  FIxe the problem Here 

        sinusoid_table = torch.stack(sinusoid_table, dim=0)

        sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])  # dim 2i
        sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])  # dim 2i+1

        posEnc = rearrange(sinusoid_table, 'l n h w  -> n h w l')


        return posEnc
    def forward(self, x):
        '''
        :param x: (b, channel, h, w, seqlen)
        :return:
        '''
        batch_size, channel, h, w, seqlen = x.shape
        PosEncoding = self.pos_table.clone().detach()
        return x + PosEncoding[None,:,:,:,:].repeat(batch_size, 1, 1, 1, 1)
