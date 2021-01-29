import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import nets


class MyModel(object):
    def __init__(self, device, train_loader, test_loader, config):
        super(MyModel, self).__init__()
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        self.net = nets.MyNetwork().to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'])

    def train(self):
        with tqdm(range(1, self.config['epochs']+1), desc='training') as pbar1:
            for epoch in pbar1:
                with tqdm(enumerate(self.train_loader), desc='iterating') as pbar2:
                    for batch_idx, sample in pbar2:
                        self.net.train()

                        img, label = sample['img'].to(self.device), sample['label'].to(self.device)

                        self.optimizer.zero_grad()

                        output = self.net(img)

                        loss = 0

                        loss.backward()
                        self.optimizer.step()

                        pbar2.set_postfix(loss = loss.item())

                pbar1.set_postfix(val_loss = 0)
                self.save_weights(self.config['weights_dir'])


    def save_weights(self, weights_dir):
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        weights_path = os.path.join(weights_dir, 'weights')
        state = {
                'weights': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                }

        torch.save(state, weights_path)


    def load_weights(self):
        weights_dir = self.config['weights_dir']
        if not os.path.exists(weights_dir):
            raise ValueError("'" + weights_dir + "' does not exist.")

        weights_path = os.path.join(weights_dir, 'weights')

        state = torch.load(weights_path, map_location=self.device)
        self.net.load_state_dict(state['weights'])
        self.optimizer.load_state_dict(state['optimizer'])


    def test(self):
        self.net.eval()
        with torch.no_grad():
            pass


    def visualize(self):
        self.net.eval()
        with torch.no_grad():
            pass
