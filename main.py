import os
import argparse
import numpy
import torch
import torchvision
import yaml

import datasets
import models


def check_yml(d):
    try:
        assert os.path.isfile(d)
        return d
    except Exception:
        raise argparse.ArgumentTypeError(
            "yml file {} cannot be located.".format(d))

def main():
    parser = argparse.ArgumentParser(description='Project Name')
    parser.add_argument('mode', nargs="?",
                        choices=["train", "test", "visualize"])
    parser.add_argument('yml', nargs='?',
                        help='yml file name')

    args = parser.parse_args()

    check_yml(args.yml)
    with open(args.yml) as f:
        config = yaml.load(f, yaml.SafeLoader)

    # load param from yaml file
    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    numpy.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # preprocessing 
    transform = torchvision.transforms.Compose([
        ])

    train_dataset = datasets.MyDataset(transform = transform)
    test_dataset = datasets.MyDataset(transform = transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, **kwargs)

    model = models.MyModel(device = device, train_loader = train_loader, test_loader = test_loader)

    if args.mode=='train':
        model.train(epochs = config['epochs'], lr=config['lr'])


    elif args.mode=='test':
        pass


    elif args.mode=='visualize':
        pass


if __name__ == '__main__':
    main()
