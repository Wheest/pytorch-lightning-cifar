"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl

import torchvision
import torchvision.transforms as transforms

import multiprocessing

import os
import argparse

import models
from utils import progress_bar

MODELS = {
    "densenet_cifar": models.densenet_cifar,
    "densenet_121": models.DenseNet121,
    "densenet_161": models.DenseNet169,
    "densenet_169": models.DenseNet169,
    "densenet_201": models.DenseNet201,
    "dla": models.DLA,
    "dla_simple": models.SimpleDLA,
    "dpn26": DPN26,
    "dpn92": DPN92,
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
    "resnet18": model.ResNet18,
    "resnet34": model.ResNet34,
    "resnet50": model.ResNet50,
    "resnet101": model.ResNet101,
    "resnet152": model.ResNet152,
    "resnext29_2x64d": models.ResNeXt29_2x64d,
    "resnext29_4x64d": models.ResNeXt29_4x64d,
    "resnext29_8x64d": models.ResNeXt29_8x64d,
    "resnext29_32x4d": models.ResNeXt29_32x4d,
    "senet18": models.SENet18,
    "shufflenetv1_g2": models.ShuffleNetV1_G2,
    "shufflenetv1_g3": models.ShuffleNetV1_G3,
    "shufflenetv2_0_5": models.ShuffleNetV2_0_5,
    "shufflenetv2_1_0": models.ShuffleNetV2_1_0,
    "shufflenetv2_1_5": models.ShuffleNetV2_1_5,
    "shufflenetv2_2_0": models.ShuffleNetV2_2_0,
    "vgg11": models.Vgg11,
    "vgg13": models.Vgg13,
    "vgg16": models.Vgg16,
    "vgg19": models.Vgg19,
}
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def setup_dataset(num_workers: int = multiprocessing.cpu_count()):
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return trainloader, testloader


def get_model(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Model `{model_name}` not currently available")
    model = MODELS[model_name]()
    return model


def main(args):
    model = get_model(args.model_name)
    model.learning_rate = args.lr
    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=10,
    )
    train_loader, test_loader = setup_dataset()
    trainer.fit(model, train_dataloaders=train_loader)
    print("Finished fitting")
    trainer.test(dataloaders=test_dataloader)


def misc():
    # Model
    print("==> Building model..")
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    # net = net.to(device)

    trainer = pl.Trainer(max_epochs=1)

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training
    def train(epoch):
        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        # Save checkpoint.
        acc = 100.0 * correct / total
        if acc > best_acc:
            print("Saving..")
            state = {
                "net": net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/ckpt.pth")
            best_acc = acc

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        nargs="+",
        choices=["all"] + list(MODELS.keys()),
        help="Model architecture to use",
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument(
        "--num_workers",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Number of workers for dataloader",
    )
    args = parser.parse_args()
    main(args)
