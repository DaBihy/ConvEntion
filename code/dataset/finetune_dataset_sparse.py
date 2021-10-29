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
import json 
import random

class FinetuneDataset(Dataset):
    def __init__(self, data_path, labelsPath, classes=None, seq_len=None):
        """
        :param file_path: fine-tuning file path
        :param seq_len: padded sequence length
        """
        self.seq_len = seq_len
        self.classes = classes
        self.Data = glob.glob(data_path)
        self.labels = json.load(open(labelsPath))
        self.TS_num = len(self.Data) 

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):

        sequencePath = self.Data[item]
        # sequence =  rearrange( np.load(sequencePath), 'l n h w -> n h w l',l=24,n=10,h=32)
        sequence =  np.load(sequencePath)

        new_seq = []
        bands = []
        for i in range(24):
            sparse = random.randint(4, 10)
            frames = random.sample(range(10), sparse)
            frames.sort()
            bands+=frames
            for j in frames:
                new_seq.append(sequence[i,j])
        BandedSeq =np.expand_dims(np.stack(new_seq,axis=-1),0)
        DaBands = np.array(bands)+1
        start_ind = randint(0, len(new_seq) - self.seq_len)
        seqId =  sequencePath.split('/')[-1].split('.')[0]
        SeqLabel = np.array(self.classes[self.labels['label_19class'][seqId]], dtype=int)
        BandedSeq = BandedSeq.astype(np.int)
        # print(f"Isze prepro {DaBands[...,start_ind:self.seq_len+start_ind].shape}")
        output = {"bert_input": BandedSeq[...,start_ind:self.seq_len+start_ind],
                  "class_label": SeqLabel,
                  "bands": DaBands[start_ind:self.seq_len + start_ind]
                  }

        return {key: torch.from_numpy(np.asarray(value)) for key, value in output.items()}

