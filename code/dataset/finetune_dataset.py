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

class FinetuneDataset(Dataset):
    def __init__(self, data_path, labelsPath, classes=None, seq_len=None):
        """
        :param file_path: fine-tuning file path
        :param seq_len: padded sequence length
        """
        # self.seq_len = seq_len
        self.classes = classes
        self.Data = glob.glob(data_path)
        self.labels = json.load(open(labelsPath))
        self.TS_num = len(self.Data) 

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):

        sequencePath = self.Data[item]
        sequence =  rearrange( np.load(sequencePath), 'l n h w -> n h w l',l=24,n=10,h=32)
        seqId =  sequencePath.split('/')[-1].split('.')[0]
        SeqLabel = np.array(self.classes[self.labels['label_19class'][seqId]], dtype=int)
        sequence = sequence.astype(np.int)
        output = {"bert_input": sequence,
                  "class_label": SeqLabel
                  }

        return {key: torch.from_numpy(np.asarray(value)) for key, value in output.items()}

