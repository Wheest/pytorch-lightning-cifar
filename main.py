"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from tqdm import tqdm
import json

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

import torchvision
import torchvision.transforms as transforms

import multiprocessing

import os
import argparse

import models
from utils import progress_bar

pl.seed_everything(7)
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

MODELS = {
    "densenet_cifar": models.densenet_cifar,
    "densenet_121": models.DenseNet121,
    "densenet_161": models.DenseNet169,
    "densenet_169": models.DenseNet169,
    "densenet_201": models.DenseNet201,
    "dla": models.DLA,
    "dla_simple": models.SimpleDLA,
    "dpn26": models.DPN26,
    "dpn92": models.DPN92,
    "efficientnetb0": models.EfficientNetB0,
    "googlenet": models.GoogLeNet,
    "lenet": models.LeNet,
    "mobilenetv1": models.MobileNetV1,
    "mobilenetv2": models.MobileNetV2,
    "pnasnet_a": models.PNASNetA,
    "pnasnet_b": models.PNASNetB,
    "preactresnet18": models.PreActResNet18,
    "preactresnet34": models.PreActResNet34,
    "preactresnet50": models.PreActResNet50,
    "preactresnet101": models.PreActResNet101,
    "preactresnet152": models.PreActResNet152,
    "regnetx_200mf": models.RegNetX_200MF,
    "regnetx_400mf": models.RegNetX_400MF,
    "regnety_400mf": models.RegNetY_400MF,
    "resnet18": models.ResNet18,
    "resnet34": models.ResNet34,
    "resnet50": models.ResNet50,
    "resnet101": models.ResNet101,
    "resnet152": models.ResNet152,
    "resnext29_2x64d": models.ResNeXt29_2x64d,
    "resnext29_4x64d": models.ResNeXt29_4x64d,
    "resnext29_8x64d": models.ResNeXt29_8x64d,
    "resnext29_32x4d": models.ResNeXt29_32x4d,
    "senet18": models.SENet18,
    # "shufflenetv1_g2": models.ShuffleNetV1_G2, # currently don't work, PRs accepted
    # "shufflenetv1_g3": models.ShuffleNetV1_G3,
    "shufflenetv2_0_5": models.ShuffleNetV2_0_5,
    "shufflenetv2_1_0": models.ShuffleNetV2_1_0,
    "shufflenetv2_1_5": models.ShuffleNetV2_1_5,
    "shufflenetv2_2_0": models.ShuffleNetV2_2_0,
    "vgg11": models.Vgg11,
    "vgg13": models.Vgg13,
    "vgg16": models.Vgg16,
    "vgg19": models.Vgg19,
}


def setup_dataset(num_workers: int = multiprocessing.cpu_count()):
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir="data/",
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    return cifar10_dm


def get_model(model_name, lr):
    if model_name not in MODELS.keys():
        raise ValueError(f"Model `{model_name}` not currently available")
    return MODELS[model_name](lr=lr)


def main(args):

    if "all" in args.model_arch:
        args.model_arch = list(MODELS.keys())

    pbar = tqdm(args.model_arch)
    for m in pbar:
        pbar.set_description(f"Training model {m}")
        model = get_model(m, args.lr)

        # configure data loader
        cifar10_dm = setup_dataset(args.num_workers)
        model.datamodule = cifar10_dm

        # Initialize a trainer
        logger = pl.loggers.TensorBoardLogger("tb_logs", name=m)

        trainer = pl.Trainer(
            gpus=AVAIL_GPUS,
            max_epochs=args.epochs,
            callbacks=[pl.callbacks.LearningRateMonitor(logging_interval="step")],
            logger=logger,
        )

        # train
        trainer.fit(model, cifar10_dm)

        # test
        trainer.test(datamodule=cifar10_dm)

        # save final metrics to file
        with open(
            os.path.join(
                trainer.logger.save_dir,
                m,
                f"version_{trainer.logger.version}",
                "final_metrics.json",
            ),
            "w",
        ) as f:
            metrics = {k: v.item() for k, v in trainer.logged_metrics.items()}
            metrics["trainable_parameters"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            metrics["total_parameters"] = sum(p.numel() for p in model.parameters())
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "--model_arch",
        required=True,
        type=str,
        nargs="+",
        choices=["all"] + list(MODELS.keys()),
        help="Model architecture to use",
    )
    parser.add_argument("--lr", default=0.05, type=float, help="learning rate")
    parser.add_argument(
        "--epochs", default=200, type=float, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--num_workers",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Number of workers for dataloader",
    )
    args = parser.parse_args()
    main(args)
