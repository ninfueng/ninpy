import logging
import multiprocessing
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import test, train, trainv2, warmup

from ninpy.torch2.models import resnet20
from ninpy.torch2.datasets.augment import get_cifar10_transforms
from ninpy.torch2.datasets.toys import load_toy_dataset

from ninpy.torch2 import (
    load_model,
    ninpy_setting,
    save_model,
    tensorboard_hparams,
    tensorboard_models,
)

if __name__ == "__main__":
    hparams, exp_pth, writer = ninpy_setting(
        "cifar10", "hyper.yaml", benchmark=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transforms, test_transforms = get_cifar10_transforms()
    train_loader, test_loader = load_toy_dataset(
        hparams.train_batch,
        hparams.test_batch,
        multiprocessing.cpu_count(),
        "cifar10",
        train_transforms=train_transforms,
        test_transforms=test_transforms,
    )
    model = resnet20()
    writer.add_graph(model, torch.zeros(1, 3, 32, 32))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(hparams.init_lr),
        weight_decay=float(hparams.weight_decay),
    )

    model = model.to(device)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=hparams.step_size, gamma=hparams.step_down_rate
    )

    best_acc = 0.0
    pbar = tqdm(range(hparams.epochs))
    for epoch in pbar:
        trainv2(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            epoch,
            writer,
        )
        test_acc = test(model, device, test_loader, criterion, epoch, writer)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(
                os.path.join(
                    exp_pth, f"{test_acc:.4f}".replace(".", "_") + ".pth"
                ),
                model,
                optimizer,
                save_epoch=hparams.save_epoch,
                epoch=epoch,
            )
        tensorboard_models(writer, model, epoch)

    logging.info(f"Best test accuracy: {best_acc}")
    metric_dict = {"best_acc": best_acc}
    tensorboard_hparams(
        writer, hparam_dict=hparams.to_dict(), metric_dict=metric_dict
    )
    metric_dict.update(hparams)
    # basic_notify(metric_dict)
