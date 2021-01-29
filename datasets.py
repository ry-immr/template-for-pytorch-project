import os
import skimage.io
import numpy
import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.dataset = [[0, 0]]

        self.transform = transform


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = {'img': self.dataset[idx][0].copy(), 'label': self.dataset[idx][1].copy()}

        if self.transform:
            sample = self.transform(sample)

        return sample
