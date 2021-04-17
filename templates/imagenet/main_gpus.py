"""
TODO: Consider adding SyncBatchNorm to low batch size training.
"""
import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from apex import amp
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet18
from utils import test, train, warmup

from ninpy.datasets import get_imagenet_loaders
from ninpy.notify import basic_notify
from ninpy.torch_utils import (
    load_model,
    ninpy_setting,
    save_model,
    tensorboard_hparams,
    tensorboard_models,
)

# https://stackoverflow.com/questions/58271635/in-distributed-computing-what-are-world-size-and-rank
# inside of main_worker.

def worker(rank, hparams):
    assert isinstance(rank, int)
    assert rank < hparams.world_size

    global best_acc
    verbose = rank == 0
    hparams.train_batch = int(hparams.train_batch/ hparams.world_size)
    hparams.num_workers = int(hparams.num_workers / hparams.world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend=hparams.backend,
        init_method="tcp://224.66.41.62:23456",
        rank=rank,
        world_size=hparams.world_size,
    )
    train_loader, test_loader = get_imagenet_loaders(
        hparams.dataset_dir, hparams.train_batch, hparams.num_workers, distributed=True
    )

    model = resnet18(pretrained=False)
    if verbose:
        writer.add_graph(model, torch.zeros(1, 3, 224, 224))
    model = model.to(rank)

    criterion = nn.CrossEntropyLoss(reduction="mean").to(rank)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(hparams.init_lr),
        weight_decay=float(hparams.weight_decay),
    )

    model, optimizer = amp.initialize(
        model, optimizer, opt_level=hparams.opt_lv, verbosity=1
    )

    model = DistributedDataParallel(model, device_ids=[rank])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=hparams.step_size, gamma=hparams.step_down_rate
    )

    best_acc = 0.0
    pbar = range(hparams.epochs)
    for epoch in pbar:
        train(
            model, rank, train_loader, optimizer, criterion, epoch, writer, verbose
        )
        test_acc = test(model, rank, test_loader, criterion, epoch, writer, verbose)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            print(best_acc)
            if verbose:
                pbar.set_description(f"Best test acc: {best_acc.item():.4f}.")
                save_model(
                    os.path.join(exp_pth, f"{test_acc:.4f}".replace(".", "_") + ".pth"),
                    model,
                    optimizer,
                    save_epoch=hparams.save_epoch,
                    epoch=epoch,
                )
        if verbose:
            tensorboard_models(writer, model, epoch)

    if verbose:
        logging.info(f"Best test accuracy: {best_acc}")
        metric_dict = {"best_acc": best_acc}
        tensorboard_hparams(writer, hparam_dict=hparams.to_dict(), metric_dict=metric_dict)

        metric_dict.update(hparams.to_dict())
        basic_notify(metric_dict)


if __name__ == "__main__":
    best_acc = 0
    hparams, exp_pth, writer = ninpy_setting("imagenet", "hyper.yaml", benchmark=True)
    mp.spawn(worker, args=(hparams,), nprocs=hparams.world_size, join=True)
    dist.destroy_process_group()
