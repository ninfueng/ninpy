import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from apex import amp
from torchvision.models import resnet18
from tqdm import tqdm
from utils import test, train, warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ninpy.datasets import get_imagenet_loaders
from ninpy.notify import basic_notify
from ninpy.torch_utils import (
    load_model,
    ninpy_setting,
    save_model,
    tensorboard_hparams,
    tensorboard_models,
)
from torch.utils.data.dataloader import DataLoader
from segnet import SegNet
from ninpy.datasets.camvid import Camvid
from ninpy.losses import class_weights

if __name__ == "__main__":
    hparams, exp_pth, writer = ninpy_setting("imagenet", "hyper.yaml", benchmark=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = A.Compose([A.Resize(360, 480), A.Normalize(), ToTensorV2()])
    train_dataset = Camvid(hparams.dataset_dir, "train", transforms)
    test_dataset = Camvid(hparams.dataset_dir, "test", transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
    )

    model = SegNet(11, 3)
    writer.add_graph(model, torch.zeros(1, 3, 360, 480))
    model = model.to(device)
    cls_w = class_weights(train_dataset)

    criterion = nn.CrossEntropyLoss(
        reduction="mean", ignore_index=11).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(hparams.init_lr),
        weight_decay=float(hparams.weight_decay),
    )

    model, optimizer = amp.initialize(
        model, optimizer, opt_level=hparams.opt_lv, verbosity=1
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=hparams.step_size, gamma=hparams.step_down_rate
    )

    best_acc = 0.0
    pbar = tqdm(range(hparams.epochs))
    for epoch in pbar:
        train(
            model, device, train_loader, optimizer, criterion, epoch, writer,
        )
        test_acc = test(model, device, test_loader, criterion, epoch, writer)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            pbar.set_description(f"Best test acc: {best_acc.item():.4f}.")
            save_model(
                os.path.join(exp_pth, f"{test_acc:.4f}".replace(".", "_") + ".pth"),
                model,
                optimizer,
                save_epoch=hparams.save_epoch,
                epoch=epoch,
            )
        tensorboard_models(writer, model, epoch)

    logging.info(f"Best test accuracy: {best_acc}")
    metric_dict = {"best_acc": best_acc}
    tensorboard_hparams(writer, hparam_dict=hparams.to_dict(), metric_dict=metric_dict)
    metric_dict.update(hparams.to_dict())
    basic_notify(metric_dict)
