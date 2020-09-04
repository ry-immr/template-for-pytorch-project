import os
import skimage.io
import numpy
import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.dataset = []

        self.transform = transform


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = {'img': self.dataset[idx][0], 'label': self.dataset[idx][1]}

        if self.transform:
            sample = self.transform(sample)

        return sample
