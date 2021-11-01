from torch.utils.data import Dataset
import torch
import numpy as np
import os
import math
import copy
import random
import glob
from random import randint
from einops import rearrange
import pandas as pd
import pickle5 as pickle
import torch
from astropy.table import Table
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import skimage.transform 


def RandomRotation(clip, degrees):
    angle = random.uniform(-degrees, degrees)
    rotated = [skimage.transform.rotate(img[0], angle) for img in clip]
    Rot_Clip = np.expand_dims(np.stack(rotated,axis=0), axis=1)
    return Rot_Clip


def RandomDrop(clip, filters, seq_len):
    if seq_len>= clip.shape[0]:
        return clip, filters
    else:
        num_of = clip.shape[0] - seq_len
        rand_num = randint(num_of//2, num_of)
        randomlist = random.sample(range(0, clip.shape[0]), rand_num)
        new_clip = np.delete(clip, np.array(randomlist, dtype=int) , 0)
        new_filters = np.delete(filters, np.array(randomlist, dtype=int) , 0)
        
        return new_clip, new_filters

def SeqFlip(clip):
    rand_num = randint(0, 1)
    rotated = [np.flip(img[0],rand_num) for img in clip]
    Rot_Clip = np.expand_dims(np.stack(rotated,axis=0), axis=1)
    return Rot_Clip


class FinetuneDataset(Dataset):
    # subset can be: 'train', 'val', 'test'
    def __init__(self, path_dat, meta_path , classes=None, seq_len=24, list_IDs=None, Transform=False):
        """
        :param file_path: fine-tuning file path
        :param seq_len: padded sequence length
        """
        self.seq_len = seq_len
        self.classes = classes

        if Transform:
            lis = list_IDs
            random.shuffle(lis)
            self.list_IDs = list_IDs + lis
            self.TS_num = len(self.list_IDs)
            self.half_item = self.TS_num//2
        else:
            self.list_IDs = list_IDs 
            self.TS_num = len(self.list_IDs)
            self.half_item = self.TS_num

        self.path_dat = path_dat
       

        self.meta_path = Table.read(meta_path)

    def __getitem__(self, item):
        trans_id = self.list_IDs[item]
        trans_id = str(trans_id)
        try:
            while(len(trans_id) < 6) :
                trans_id = '0'+trans_id
            path_trans_dat = self.path_dat + "sdss_" + trans_id + '_32pix.pickle'
            with open(path_trans_dat, "rb") as fh:
                object  = pickle.load(fh)
        except:
            return None

        if object['images'].shape[0]<self.seq_len and item > self.half_item:
            return None
            
        if object['label'] == 'Unknown':
            return None

      
        ebv = 0
        clip, filters = RandomDrop(object['images'], object['filter'], self.seq_len)
        ts_length = clip.shape[0]
        start_ind = 0
        if ts_length > self.seq_len :
            start_ind = randint(0, ts_length - self.seq_len)
            ts_length = self.seq_len
        
        BandedSeq  = np.zeros((self.seq_len, 1 ,32,32))

        if object['label'] == 'Unknown':
            return None

        try:
            BandedSeq[:ts_length,0] = clip[start_ind:ts_length+start_ind]
        except:
            print(self.Data[item])
            print(clip.shape)
             
        type_aug = randint(0, 1)
        if type_aug == 0:
            BandedSeq = RandomRotation(BandedSeq, degrees=20)
        else:
            BandedSeq = SeqFlip(BandedSeq)

        SeqLabel =  np.array(self.classes[object['label']], dtype=int)
        DaBands = np.zeros((self.seq_len, ))
        DaBands[:ts_length] = filters[start_ind:ts_length+start_ind]
       
        std= np.array([1.87221809e-10, 7.85509139e-11, 1.42964490e-10, 2.15037795e-10, 3.89587566e-10])
        mean = np.array([1.17335142e-11, 2.41511189e-11, 5.53039069e-11, 5.17518492e-11,9.30239258e-11])
        for i in range(ts_length):
            BandedSeq[i] = (BandedSeq[i]-mean[int(DaBands[i])])/std[int(DaBands[i])]
        BandedSeq =  rearrange( BandedSeq, 'l n h w  ->n h w l',l=self.seq_len,n=1,h=32 ,w=32)
       

        #   sepcify the weight 1 if the objecti specto confirmed 0.5 if not 
        weight_obt = 0.7
        confirmed_obt = ['AGN', 'SNIa', 'SNIa?', 'SLSN', 'SNIc', 'Variable', 'SNII', 'SNIb']
        if object['label'] in confirmed_obt:
            weight_obt = 1

        output = {"bert_input": BandedSeq,
                  "class_label": SeqLabel,
                  "bands": DaBands,
                  "ebv": ebv, 
                  "weight": float(weight_obt)
                  }

        return {key: torch.from_numpy(np.asarray(value)) for key, value in output.items()}

    def __len__(self):
        return self.TS_num