import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import nets


class MyModel(object):
    def __init__(self, device, train_loader, test_loader):
        super(MyModel, self).__init__()
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.net = nets.MyNetwork().to(self.device)


    def train(self, epochs, lr):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.net.train()

        with tqdm(range(1, epochs+1), desc='training') as pbar1:
            for epoch in pbar1:
                with tqdm(enumerate(self.train_loader), desc='iterating') as pbar2:
                    for batch_idx, sample in pbar2:
                        img, label = sample['img'].to(self.device), sample['label'].to(self.device)

                        optimizer.zero_grad()

                        output = self.net(img)

                        loss = 0

                        loss.backward()
                        optimizer.step()

                        pbar2.set_postfix(loss = loss.item())

                pbar1.set_postfix(val_loss = val_loss)


    def save_weights(self, file_path):
        torch.save(self.net.state_dict(), file_path)


    def load_weights(self, file_path):
        self.net.load_state_dict(torch.load(file_path, map_location=self.device))


    def test(self):
        self.net.eval()
        with torch.no_grad():
            pass


    def visualize(self):
        self.net.eval()
        with torch.no_grad():
            pass
