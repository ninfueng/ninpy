#!usr/bin/env python
import logging
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from ninpy.common import RunningAverage
from ninpy.torch_utils import set_warmup_lr
from prefetch_generator import BackgroundGenerator


def warmup(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    initial_lr: float,
    warmup_epochs: int,
) -> None:
    avgloss, avgacc = RunningAverage(), RunningAverage()
    model.train()
    for w in range(warmup_epochs):
        for idx, (data, target) in BackgroundGenerator(enumerate(train_loader)):
            set_warmup_lr(
                initial_lr, warmup_epochs, train_loader, optimizer, idx, w, False
            )
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            batch_size = target.shape[0]
            loss = criterion(output, target)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            acc = (output.argmax(-1) == target).float().sum()
            avgloss.update(loss, batch_size)
            avgacc.update(acc, batch_size)
        logging.info(f"Warmup epoch {w} Acc: {avgacc():.4f} Loss: {avgloss():.4f}")


def train(model, device, train_loader, optimizer, criterion, epoch, writer=None):
    avgloss, avgacc = RunningAverage(), RunningAverage()
    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        batch_size = target.shape[0]
        loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        acc = (output.argmax(-1) == target).float().sum()
        avgloss.update(loss, batch_size)
        avgacc.update(acc, batch_size)

    logging.info(f"Train epoch {epoch} Acc: {avgacc():.4f} Loss: {avgloss():.4f}")
    if writer is not None:
        writer.add_scalar("train_acc", avgacc(), epoch)
        writer.add_scalar("train_loss", avgloss(), epoch)


def test(model, device, test_loader, criterion, epoch: int, writer=None) -> float:
    avgloss, avgacc = RunningAverage(), RunningAverage()

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            batch_size = target.shape[0]
            output = model(data)

            loss = criterion(output, target)
            acc = (output.argmax(-1) == target).float().sum()
            avgloss.update(loss, batch_size)
            avgacc.update(acc, batch_size)

    logging.info(f"Test epoch {epoch} Acc: {avgacc():.4f} Loss: {avgloss():.4f}")
    if writer is not None:
        writer.add_scalar("test_acc", avgacc(), epoch)
        writer.add_scalar("test_loss", avgloss(), epoch)
    return avgacc()
