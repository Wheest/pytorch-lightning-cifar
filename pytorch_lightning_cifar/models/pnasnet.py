"""PNASNet in PyTorch.

Paper: Progressive Neural Architecture Search
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import OneCycleLR


class SepConv(nn.Module):
    """Separable Convolution."""

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
            groups=in_planes,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(CellA, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride == 2:
            self.conv1 = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride == 2:
            y2 = self.bn1(self.conv1(y2))
        return F.relu(y1 + y2)


class CellB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(CellB, self).__init__()
        self.stride = stride
        # Left branch
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        # Right branch
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        if stride == 2:
            self.conv1 = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_planes)
        # Reduce channels
        self.conv2 = nn.Conv2d(
            2 * out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride == 2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        # Concat & reduce channels
        b1 = F.relu(y1 + y2)
        b2 = F.relu(y3 + y4)
        y = torch.cat([b1, b2], 1)
        return F.relu(self.bn2(self.conv2(y)))


class PNASNet(pl.LightningModule):
    def __init__(self, cell_type, num_cells, num_planes, lr=0.05):
        super(PNASNet, self).__init__()

        self.save_hyperparameters()

        self.in_planes = num_planes
        self.cell_type = cell_type

        self.conv1 = nn.Conv2d(
            3, num_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_planes)

        self.layer1 = self._make_layer(num_planes, num_cells=6)
        self.layer2 = self._downsample(num_planes * 2)
        self.layer3 = self._make_layer(num_planes * 2, num_cells=6)
        self.layer4 = self._downsample(num_planes * 4)
        self.layer5 = self._make_layer(num_planes * 4, num_cells=6)

        self.linear = nn.Linear(num_planes * 4, 10)

    def _make_layer(self, planes, num_cells):
        layers = []
        for _ in range(num_cells):
            layers.append(self.cell_type(self.in_planes, planes, stride=1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes):
        layer = self.cell_type(self.in_planes, planes, stride=2)
        self.in_planes = planes
        return layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 8)
        out = self.linear(out.view(out.size(0), -1))
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


def PNASNetA(lr=0.05):
    return PNASNet(CellA, num_cells=6, num_planes=44, lr=lr)


def PNASNetB(lr=0.05):
    return PNASNet(CellB, num_cells=6, num_planes=32, lr=lr)


def test():
    net = PNASNetB()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)


# test()
