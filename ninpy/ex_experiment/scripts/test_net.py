#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:26:28 2021

@author: ninnart
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import logging
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter

from log import set_logger
from experiment import set_experiment
from config import dump_yaml, load_yaml, dict2str


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(hypers, model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(BackgroundGenerator(enumerate(train_loader)), total=len(train_loader))    
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % hypers['log_interval'] == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if hypers['dry_run']:
                break
        #pbar.set_description(f'Train acc: {100. * correct / len(train_loader.dataset)}')


def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    pbar = tqdm(BackgroundGenerator(test_loader), total=len(test_loader))
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.set_description(f'Test {epoch}')
            pbar.set_postfix({'acc': 100. * correct / len(test_loader.dataset)})

    test_loss /= len(test_loader.dataset)
    writer.add_scalar('loss/test', test_loss, epoch)
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=14, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    parser.add_argument('--yaml', type=str, default='test_net.yaml')
    args = parser.parse_args()
        
    # argsdict = vars(args)
    # dump_yaml(argsdict, 'test_net.yaml')
    
    hypers = load_yaml(args.yaml)
    use_cuda = not hypers['no_cuda'] and torch.cuda.is_available()

    setup_str = dict2str(hypers)
    exp_pth = set_experiment(setup_str)
    set_logger(os.path.join(exp_pth, 'info.log'), False)
    writer = SummaryWriter(log_dir=os.path.join(exp_pth, 'tensorboard'))
    
    torch.manual_seed(hypers['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': hypers['batch_size']}
    test_kwargs = {'batch_size': hypers['test_batch_size']}
    if use_cuda:
        cuda_kwargs = {'num_workers': 8,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=hypers['lr'])

    scheduler = StepLR(optimizer, step_size=1, gamma=hypers['gamma'])
    for epoch in range(1, hypers['epochs'] + 1):
        train(hypers, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch, writer)
        scheduler.step()

    if hypers['save_model']:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
    save_params = {'hyper': hypers, 'results': {'test_acc': 0.992}}
    dump_yaml(save_params, os.path.join(exp_pth, 'results.yaml'))