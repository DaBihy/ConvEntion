import os
import random
import numpy as np
import torch
import pickle5 as pickle
import skimage.transform
from torch.utils.data import Dataset
from astropy.table import Table
from einops import rearrange
from random import randint

def random_rotation(clip, degrees):
    """Applies a random rotation to each image in the clip within the specified degree range."""
    angle = random.uniform(-degrees, degrees)
    rotated = [skimage.transform.rotate(img[0], angle) for img in clip]
    return np.expand_dims(np.stack(rotated, axis=0), axis=1)

def random_drop(clip, filters, seq_len, transform):
    """Randomly drops frames from the clip to reduce its length to seq_len."""
    if seq_len >= clip.shape[0] or not transform:
        return clip, filters

    num_to_drop = clip.shape[0] - seq_len
    drop_indices = random.sample(range(clip.shape[0]), randint(num_to_drop // 2, num_to_drop))
    return np.delete(clip, drop_indices, axis=0), np.delete(filters, drop_indices, axis=0)

def seq_flip(clip):
    """Flips the clip horizontally or vertically based on a random choice."""
    flip_axis = randint(0, 1)
    return np.expand_dims(np.stack([np.flip(img[0], flip_axis) for img in clip], axis=0), axis=1)

class ZTFDataset(Dataset):
    def __init__(self, path_dat, meta_path, classes=None, seq_len=24, list_IDs=None, transform=False):
        """
        Initializes the ZTF Dataset.

        :param path_dat: Path to the dataset.
        :param meta_path: Path to the metadata.
        :param classes: Class labels.
        :param seq_len: Length of the time series sequence.
        :param list_IDs: List of IDs in the dataset.
        :param transform: Boolean flag to apply transformations.
        """
        self.seq_len = seq_len
        self.classes = classes
        self.list_IDs = list_IDs
        self.TS_num = len(self.list_IDs)
        self.path_dat = path_dat
        self.transform = transform
        self.meta_path = Table.read(meta_path)

    def __getitem__(self, item):
        trans_id = str(self.list_IDs[item]).zfill(6)
        path_trans_dat = os.path.join(self.path_dat, f'{trans_id}_32pix.pickle')

        try:
            with open(path_trans_dat, "rb") as fh:
                object = pickle.load(fh)
        except:
            return None
            
        if object['label'] == 'Unknown':
            return None

        filter_map = {'r': 0, 'g': 1, 'i': 2}
        filters = np.array([filter_map[fil] for fil in object['filter']])
        clip, filters = random_drop(np.nan_to_num(object['image']), filters, self.seq_len, self.transform)

        ts_length = min(clip.shape[0], self.seq_len)
        start_ind = randint(0, clip.shape[0] - ts_length) if clip.shape[0] > self.seq_len else 0

        banded_seq = np.zeros((self.seq_len, 1, 32, 32))
        banded_seq[:ts_length, 0] = clip[start_ind:ts_length + start_ind]

        bands = np.zeros(self.seq_len)
        bands[:ts_length] = filters[start_ind:ts_length + start_ind]

        if self.transform:
            banded_seq = random_rotation(banded_seq, 15) if randint(0, 1) == 0 else seq_flip(banded_seq)

        std = np.array([71.14060526, 37.02519333, 60.835895])
        mean = np.array([581.21864938, 447.12376419, 539.32994191])
        for i in range(ts_length):
            banded_seq[0, i] = (banded_seq[0, i] - mean[int(bands[i])]) / std[int(bands[i])]

        banded_seq = rearrange(banded_seq, 'a l n h w -> a n h w l', a=2, l=self.seq_len, n=1, h=32, w=32)
        seq_label = np.array(self.classes[object['label']], dtype=int)

        return {key: torch.from_numpy(value) for key, value in {"bert_input": banded_seq, "class_label": seq_label, "bands": bands}.items()}

   
