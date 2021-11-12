import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from apex import amp
from torchvision.models import resnet18
from tqdm import tqdm
from utils import test, train, warmup

from ninpy.torch2.datasets import get_imagenet_loaders
from ninpy.notify import basic_notify
from ninpy.torch2 import (
    load_model,
    ninpy_setting,
    save_model,
    tensorboard_hparams,
    tensorboard_models,
)

if __name__ == "__main__":
    hparams, exp_pth, writer = ninpy_setting(
        "imagenet", "hyper.yaml", benchmark=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_imagenet_loaders(
        hparams.dataset_dir,
        hparams.train_batch,
        hparams.num_workers,
        distributed=False,
    )

    model = resnet18(pretrained=False)
    # device_ids = [i for i in range(torch.cuda.device_count())]
    writer.add_graph(model, torch.zeros(1, 3, 224, 224))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
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

    if hparams.resume:
        from ninpy.torch2 import load_model

        load_model(hparams.resume_path, model, optimizer, None, scheduler)

    best_acc = 0.0
    pbar = tqdm(range(hparams.epochs))
    for epoch in pbar:
        train(
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
            pbar.set_description(f"Best test acc: {best_acc.item():.4f}.")
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
    metric_dict.update(hparams.to_dict())
    basic_notify(metric_dict)
