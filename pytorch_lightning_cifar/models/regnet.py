"""RegNet in PyTorch.

Paper: "Designing Network Design Spaces".

Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import OneCycleLR


class SE(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = F.relu(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    def __init__(self, w_in, w_out, stride, group_width, bottleneck_ratio, se_ratio):
        super(Block, self).__init__()
        # 1x1
        w_b = int(round(w_out * bottleneck_ratio))
        self.conv1 = nn.Conv2d(w_in, w_b, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w_b)
        # 3x3
        num_groups = w_b // group_width
        self.conv2 = nn.Conv2d(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=num_groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(w_b)
        # se
        self.with_se = se_ratio > 0
        if self.with_se:
            w_se = int(round(w_in * se_ratio))
            self.se = SE(w_b, w_se)
        # 1x1
        self.conv3 = nn.Conv2d(w_b, w_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w_out)

        self.shortcut = nn.Sequential()
        if stride != 1 or w_in != w_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(w_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.with_se:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RegNet(pl.LightningModule):
    def __init__(self, cfg, num_classes=10, lr=0.05):
        super(RegNet, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(0)
        self.layer2 = self._make_layer(1)
        self.layer3 = self._make_layer(2)
        self.layer4 = self._make_layer(3)
        self.linear = nn.Linear(self.cfg["widths"][-1], num_classes)

    def _make_layer(self, idx):
        depth = self.cfg["depths"][idx]
        width = self.cfg["widths"][idx]
        stride = self.cfg["strides"][idx]
        group_width = self.cfg["group_width"]
        bottleneck_ratio = self.cfg["bottleneck_ratio"]
        se_ratio = self.cfg["se_ratio"]

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(
                Block(self.in_planes, width, s, group_width, bottleneck_ratio, se_ratio)
            )
            self.in_planes = width
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.trainer.datamodule.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")


def RegNetX_200MF(lr=0.05):
    cfg = {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "strides": [1, 1, 2, 2],
        "group_width": 8,
        "bottleneck_ratio": 1,
        "se_ratio": 0,
    }
    return RegNet(cfg, lr=lr)


def RegNetX_400MF(lr=0.05):
    cfg = {
        "depths": [1, 2, 7, 12],
        "widths": [32, 64, 160, 384],
        "strides": [1, 1, 2, 2],
        "group_width": 16,
        "bottleneck_ratio": 1,
        "se_ratio": 0,
    }
    return RegNet(cfg, lr=lr)


def RegNetY_400MF(lr=0.05):
    cfg = {
        "depths": [1, 2, 7, 12],
        "widths": [32, 64, 160, 384],
        "strides": [1, 1, 2, 2],
        "group_width": 16,
        "bottleneck_ratio": 1,
        "se_ratio": 0.25,
    }
    return RegNet(cfg, lr=lr)


def test():
    net = RegNetX_200MF()
    print(net)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
