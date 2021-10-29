
from typing import Type
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import randint
import os
import math
import copy
import random
import glob
import torch
from einops import rearrange
import torch.nn as nn
import glob 
import json
# import skvideo.io
x = torch.randn(64 , 1, 10)
y = torch.randn(64 , 1, 10)
z = torch.cat((x,y),1)

# input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# embedding_band = nn.Embedding(300,1024, padding_idx=0)
# ban= embedding_band(input)
# DaBand = rearrange(torch.unsqueeze(x,2),'b l n (h w)  -> b n h w l', b=64,n=1,h=32,w=32,l=120)

# z = torch.zeros(x.size())
# print(z.shape)
# y = x.view(180, -1, 3, 10)
# print(y.shape)

# t = torch.tensor([[[[1, 2],[3, 4],[7, 8]],[[5, 6],[7, 8],[7, 8]],[[5, 6],[7, 8],[7, 8]]],[[[1, 2],[3, 4],[7, 8]],[[5, 6],[7, 8],[7, 8]],[[5, 6],[7, 8],[7, 8]]]])
# s = torch.tensor([[[[1, 2],[3, 4],[7, 8]],[[5, 6],[7, 8],[7, 8]]],[[[1, 2],[3, 4],[7, 8]],[[5, 6],[7, 8],[7, 8]]]])
# z = rearrange(t , 'b n h w  -> b (n w h)')
# f = rearrange(z , 'b (n w h)  -> b n h w',n=2, h=3, w=2 )
# m = nn.Linear(10, 10)
# x[...,:2] = z
# input1 = torch.randn(128, 10,20)

# input2 = torch.randn(128, 15,7,5,10)

# out = input1.view(128,10,1,20)
# print(out.shape)


# output = m(input)
# print(torch.flatten(input,start_dim=1 ).size())
# print(t.shape)
# print(t[0::2].size())

# r = [0,1,2,3,4]
# print(r[1:])
# print(x.size(2))

# def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
#     """
#     Utility function for computing output of convolutions
#     takes a tuple of (h,w) and returns a tuple of (h,w)
#     """
    
#     if type(h_w) is not tuple:
#         h_w = (h_w, h_w)
    
#     if type(kernel_size) is not tuple:
#         kernel_size = (kernel_size, kernel_size)
    
#     if type(stride) is not tuple:
#         stride = (stride, stride)
    
#     if type(pad) is not tuple:
#         pad = (pad, pad)
    
#     h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
#     w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
#     return h, w


# h, w = conv_output_shape(h_w=3, kernel_size=1, stride=1, pad=0, dilation=1)
# print(h)

# h, w = conv_output_shape(h_w=h, kernel_size=5, stride=1, pad=2, dilation=1)
# print(h)

# h, w = conv_output_shape(h_w=h, kernel_size=3, stride=1, pad=0, dilation=1)
# print(h)

# h, w = conv_output_shape(h_w=h, kernel_size=3, stride=1, pad=0, dilation=1)
 
# print(121//4)
import json 
allfiles = glob.glob('/home/barrage/anass/sentinel/S2-2017-T31TFM-PixelPatch/s2-2017-IGARSS-NNI-NPY/DATA/*.npy')
np.random.shuffle(allfiles)

# for i in range(10):
# #     # print(int(allfiles[i].split('/')[-1].split('.')[0]))
#     X = np.load(allfiles[i])
#     Y = np.load(allfiles[i])
#     Y[:20] = 0
#     X = X.flatten('F')
#     Y = Y.flatten('F')
#     print(sum(X))
#     print(sum(Y))

# classes = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:9, 12:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16}
# da_classes = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# f = open('/home/barrage/anass/sentinel/S2-2017-T31TFM-PixelPatch/s2-2017-IGARSS-NNI-NPY/META/labels.json') 
# data = json.load(f) 
# # print(type(data))
# dic = {}
# # print(data['label_19class'].values())
# valu = list(data['label_19class'].values())
# for i in da_classes:
#     count = valu.count(i)
#     dic[i] = count

# print(dic)

# L = [data['label_44class']]
# u_value = set( val for dic in L for val in dic.values())
# # print(len(u_value))
# X = np.load('/home/barrage/anass/sentinel/S2-2017-T31TFM-PixelPatch/s2-2017-IGARSS-NNI-NPY/DATA/1.npy')
# print(X.shape)
# X[1,1]=0
# print(X[1,1])
# print('--------------------------------------------')
# print(X[1,2])


# # print(len(allfiles))
# # print(data)
# list =[] # create empty list
# for val in data['label_19class'].values(): 
#   if val in list: 
#     continue 
#   else:
#     list.append(val)
# # print(len(list))
# classes = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8, 11:9, 12:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16}
# print(classes[8])

# from positional_encodings import PositionalEncodingPermute1D, PositionalEncodingPermute2D, PositionalEncodingPermute3D

# p_enc_3d = PositionalEncodingPermute3D(11)
# z = torch.zeros((20,11,5,6,4))
# print(p_enc_3d(z).shape) # (20, 11, 5, 6, 4)


# import torch
# import torch.nn as nn
# import numpy as np
# from einops import rearrange


# class PositionalEncoding(nn.Module):

#     def __init__(self, configs):
#         super(PositionalEncoding, self).__init__()
#         configs = configs
#         num_hidden = configs[1]
#         register_buffer('pos_table', _get_sinusoid_encoding_table())

#     def _get_sinusoid_encoding_table(self):
#         ''' Sinusoid position encoding table '''
#         # 
#         def get_position_angle_vec(position):

#             return_list = [torch.ones((configs[2],
#                                        configs[3]))*(position / np.power(10000, 2 * (hid_j // 2) / num_hidden)) for hid_j in range(num_hidden)]
#             return torch.stack(return_list, dim=0)
#         sinusoid_table = [get_position_angle_vec(pos_i) for pos_i in range(configs[4])]
#         # TODO FIxe the problem Here 

#         sinusoid_table = torch.stack(sinusoid_table, dim=0)

#         sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])  # dim 2i
#         sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])  # dim 2i+1

#         posEnc = rearrange(sinusoid_table, 'l n h w  -> n h w l')


#         return posEnc
#     def forward(self, x):
#         '''
#         :param x: (b, channel, h, w, seqlen)
#         :return:
#         '''
#         batch_size, channel, h, w, seqlen = x.shape
#         PosEncoding = pos_table.clone().detach()
#         print(PosEncoding[0::2])
#         return x + PosEncoding[None,:,:,:,:].repeat(batch_size, 1, 1, 1, 1)

# tensor1 = torch.randn(64, 64, 9, 9, 24)

# print(f'this is the norma')
# embedding = PositionalEncoding(configs=[64, 64, 9, 9, 24])
# x = embedding(tensor1)
# tensor2 = torch.randn(4, 5)

# print(torch.matmul(tensor1, tensor2).size())
# new_seq = [0,2,4,5]
# new_seq = np.array(new_seq)+1
# import random
# print(new_seq[...])
# frames = random.sample(range(10), 6)
# frames.sort()
# print(frames)
# print(type(frames))
# print(random.sample(range(4,10), 24))


# df = pd.read_csv('/home/barrage/anass/Kinetics700/annotations/kinetics700/validate.csv')

# print(df.head())
# print(df['youtube_id'].iloc[[0]])
# print(df.shape[0])


# 
# videodata = skvideo.io.vread("/home/barrage/anass/Kinetics700/videos/valid/biking through snow/09dQut5GJ68_49_10.mp4")
# sequence =  np.expand_dims(rearrange( videodata, 'l h w n-> n h w l',l=76,n=3,h=144,w=192),0)
# newone = torch.squeeze(nn.functional.interpolate(torch.from_numpy(sequence), size=[160,200,76])).numpy()
# danew = rearrange( newone, 'n h w l -> l h w n',l=76,n=3,h=160,w=200).astype(np.uint8)
# from PIL import Image
# Image.fromarray(danew[50]).convert("RGB").save("art.png")
# print(videodata.shape)

# pickle_file = open("/home/barrage/anass/sdss/v3/stamps_v3/sdss_000679_32pix.pickle", "rb")

# print(pickle_file)

import pandas as pd
import pickle5 as pickle
# object = pd.read_pickle(r'/home/barrage/anass/sdss/v3/stamps_v3/sdss_000679_32pix.pickle')
with open('/home/barrage/anass/sdss/v3/stamps_v3/sdss_000714_32pix.pickle', "rb") as fh:
    object = pickle.load(fh)
from astropy.table import Table

all_transients = Table.read('/home/barrage/anass/sdss/master_data.fits')
var =all_transients[all_transients['CID']==679]['RA'] 
print(float(var))
# print(np.unique(object['mjd'].astype(int)))

# sequnce  = np.zeros((104, 5))
# for mjd in np.unique(object['mjd'].astype(int)):w
#     objectid =0
#     for i in range(object['mjd'].shape[0]):
        
#         if mjd == object['mjd'].astype(int)[i]:
#             if object['filter'][i] == 0:
#                 sequnce[objectid][0]= object['flux'][i]
#             elif object['filter'][i] == 1 :
#                 sequnce[objectid][1]= object['flux'][i]
#             elif object['filter'][i] == 2 :
#                 sequnce[objectid][2]= object['flux'][i]
#             elif object['filter'][i] == 3 :
#                 sequnce[objectid][3]= object['flux'][i]
#             else :
#                 sequnce[objectid][4]= object['flux'][i]
#         objectid +=1
    
# print(sequnce)
# doy = np.zeros((40,), dtype=int)

# doy[:20] = np.unique(object['mjd'].astype(int))[:20]
# print(doy)

        

# seq_len = 10
# ts_length = data['images'].shape[0]
# start_ind = 0
# if ts_length > seq_len :
#     start_ind = randint(0, ts_length - seq_len)
#     ts_length = seq_len
# BandedSeq  = np.zeros((seq_len, 1 ,32,32))
# BandedSeq[:ts_length,0] = data['images'][start_ind:ts_length+start_ind]
# print(BandedSeq.)

# alldata = glob.glob('/home/barrage/anass/sdss/v3/stamps_v3/*.pickle')
# print(len(alldata))
# from astropy.coordinates import SkyCoord
# from dustmaps.sfd import SFDQuery
# import dustmaps.sfd
# # dustmaps.sfd.fetch()
# l = np.array([0., 90., 180.])
# b = np.array([15., 0., -15.])

# coords = SkyCoord(327.555405, +0.657569, unit='deg', frame='icrs')
# sfd = SFDQuery()
# ebv = sfd(coords)

# print(f'This is the ebv {ebv}')

var = "thistest"

print(f'print this {var}')