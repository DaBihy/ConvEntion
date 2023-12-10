import os
import random
import glob
import numpy as np
import torch
import pickle5 as pickle
from torch.utils.data import Dataset
from einops import rearrange

class SDSSDataset(Dataset):
    def __init__(self, data_path, labels_path, classes=None, seq_len=None):
        """
        Initializes the dataset for training.

        Args:
            data_path (str): Path to the data files.
            labels_path (str): Path to the labels file.
            classes (dict, optional): Mapping of class labels to integers. Defaults to None.
            seq_len (int, optional): The length to which sequences will be padded or truncated. Defaults to None.
        """
        self.seq_len = seq_len
        self.classes = classes
        self.data_files = glob.glob(data_path)
        self.TS_num = len(self.data_files)

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):
        with open(self.data_files[item], "rb") as fh:
            data_object = pickle.load(fh)

        ts_length = min(data_object['images'].shape[0], self.seq_len)
        start_ind = random.randint(0, ts_length - self.seq_len) if ts_length > self.seq_len else 0

        banded_seq = np.zeros((self.seq_len, 1, 32, 32))
        banded_seq[:ts_length, 0] = data_object['images'][start_ind:start_ind + ts_length]

        seq_label = np.array(self.classes[data_object['label']], dtype=int)
        bands = np.zeros(self.seq_len)
        bands[:ts_length] = data_object['filter'][start_ind:start_ind + ts_length]

        banded_seq = rearrange(banded_seq, 'l n h w -> n h w l', l=self.seq_len, n=1, h=32, w=32)

        output = {
            "bert_input": banded_seq * 1e+9,
            "class_label": seq_label,
            "bands": bands
        }

        return {key: torch.from_numpy(np.asarray(value)) for key, value in output.items()}

