#!usr/bin/env python
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from tqdm import tqdm

from ninpy.common import RunningAverage
from ninpy.metrics import ConfusionMatrix
from ninpy.torch2 import set_warmup_lr


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
        for idx, (data, target) in enumerate(train_loader):
            set_warmup_lr(
                initial_lr,
                warmup_epochs,
                train_loader,
                optimizer,
                idx,
                w,
                False,
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
        logging.info(
            f"Warmup epoch {w} Acc: {avgacc():.4f} Loss: {avgloss():.4f}"
        )


def train(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    epoch,
    writer=None,
    verbose=True,
):
    NUM_CLASSES = 11
    confusion_matrix = ConfusionMatrix(NUM_CLASSES)
    avgloss = RunningAverage()

    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        batch_size = target.shape[0]
        loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        avgloss.update(loss, batch_size)
        pred = output.data.cpu().numpy().argmax(1)
        target = target.cpu().numpy()
        confusion_matrix.update(pred, target)

    if verbose:
        miou = confusion_matrix.miou_score()
        logging.info(
            f"Train epoch {epoch} mIoU: {miou:.4f} Loss: {avgloss():.4f}"
        )
        if writer is not None:
            writer.add_scalar("train/loss", avgloss(), epoch)
            writer.add_scalar("train/miou", miou, epoch)


def test(
    model,
    device,
    test_loader,
    criterion,
    epoch: int,
    writer=None,
    verbose=True,
) -> float:
    NUM_CLASSES = 11
    confusion_matrix = ConfusionMatrix(NUM_CLASSES)
    avgloss = RunningAverage()

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = target.shape[0]
            output = model(data)

            loss = criterion(output, target)
            avgloss.update(loss, batch_size)
            pred = output.data.cpu().numpy().argmax(1)
            target = target.cpu().numpy()
            confusion_matrix.update(pred, target)

    if verbose:
        miou = confusion_matrix.miou_score()
        logging.info(
            f"Test epoch {epoch} mIoU: {miou:.4f} Loss: {avgloss():.4f}"
        )
        if writer is not None:
            writer.add_scalar("test/loss", avgloss(), epoch)
            writer.add_scalar("test/miou", miou, epoch)
    return miou
