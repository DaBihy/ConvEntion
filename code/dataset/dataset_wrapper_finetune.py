import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#chnge the spares for normal data 
from .finetune_dataset_SDSSFewC import FinetuneDataset
import torch

np.random.seed(0)

class DataSetWrapper(object):

    def __init__(self, batch_size, valid_size, test_size, data_path, labelsPath, classes, max_length):
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.data_path = data_path
        self.max_length = max_length
        self.labelsPath = labelsPath
        self.classes = classes

    def get_data_loaders(self):
        dataset = FinetuneDataset(self.data_path, self.labelsPath, classes=self.classes, seq_len=self.max_length)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, dataset):
        num_train = len(dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split_valid = int(np.floor(self.valid_size * num_train))
        split_test = int(np.floor(self.test_size * num_train))
        split_train = split_test + split_valid
        print('training samples: %d, validation samples: %d,  test samples: %d' % (num_train-split_train, split_valid, split_test))
        train_idx, valid_idx, test_idx = indices[:num_train-split_train], indices[num_train-split_train:], indices[num_train-split_train:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  drop_last=True, num_workers=12, collate_fn=collate_fn)
        # print(f"Isze train_loader {train_loader}")

        valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  drop_last=True, num_workers=12, collate_fn=collate_fn)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=test_sampler,
                                  drop_last=True, num_workers=12, collate_fn=collate_fn)

        return train_loader, valid_loader, test_loader
