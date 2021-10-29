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

class FinetuneDataset(Dataset):
    def __init__(self, data_path, labelsPath, classes=None, seq_len=None):
        """
        :param file_path: fine-tuning file path
        :param seq_len: padded sequence length
        """
        self.seq_len = seq_len
        self.classes = classes
        self.Data = glob.glob(data_path)
        self.TS_num = len(self.Data) 

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):

        with open(self.Data[item], "rb") as fh:
            object  = pickle.load(fh)
        ts_length = object['images'].shape[0]
        start_ind = 0
        if ts_length > self.seq_len :
            start_ind = randint(0, ts_length - self.seq_len)
            ts_length = self.seq_len
         
        BandedSeq  = np.zeros((self.seq_len, 1 ,32,32))
        try:
            BandedSeq[:ts_length,0] = object['images'][start_ind:ts_length+start_ind]
        except:
            print(self.Data[item])
            print(object['images'].shape)
        # print(f'the size in data is {BandedSeq.shape}')
        SeqLabel =  np.array(self.classes[object['label']], dtype=int)
        DaBands = np.zeros((self.seq_len, ))
        DaBands[:ts_length] = object['filter'][start_ind:ts_length+start_ind]
        BandedSeq =  rearrange( BandedSeq, 'l n h w  ->n h w l',l=self.seq_len,n=1,h=32 ,w=32)
        # sequence = sequence.astype(np.int)
        output = {"bert_input": BandedSeq*1e+9,
                  "class_label": SeqLabel,
                  "bands": DaBands
                  }

        return {key: torch.from_numpy(np.asarray(value)) for key, value in output.items()}