# PyTorch Lightning CIFAR10
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg)](https://www.againstmalaria.com/perry-gibson)
[![PyPi version](https://badgen.net/pypi/v/pip/)](https://pypi.org/project/pytorch-lightning-cifar/)

## About this fork

Modified version of the [PyTorch CIFAR project](https://github.com/kuangliu/pytorch-cifar) to exploit the [PyTorch Lightning library](https://www.pytorchlightning.ai/).

In addition:
- Improvements `main.py` script, allowing you to train one or more models in a single command.
- Used PyTorch Lightning data module for the dataset (from the `lightning-bolts` package).
- Optimizer changed to use OneCycleLR scheduler.
- [`black`](https://github.com/psf/black) formatter applied to all files.
- Added more consistent config of VGG and ShuffleNetV2 models.
- Added Tensorboard logging, and a JSON file `final_metrics.json` that saves final accuracy.
- Create an installable Python package, which makes it easier to use as a dependency (but it's not required!).

This fork was developed while at University of Glasgow's [gicLAB](https://twitter.com/gic_lab).

Install the package with `pip install pytorch-lightning-cifar`, or clone this repo.

## Why is this useful?

CIFAR-10 is a small image classification dataset, which can be useful for validating research ideas (since models are smaller, and cheaper to train).
It has two key advantages over the MNIST database in that the problem is a bit harder, and the images are non-greyscale.
However, when doing research, you want to be able to have access to as much functionality as possible, without having to write a lot of boilerplate code.
That is the design philosophy of the PyTorch Lightning library, hence why combining the two together makes sense.
Note that few (if any) of these model architectures can be considered "official implementations", since the architectures have to be changed slightly to support the different data sizes.
However, I have seen these models used in research, you need merely say that the models are defined for CIFAR10 (and ideally cite this repo, see the righthand side of the GitHub page!).

## Contributing
PRs especially appreciated for features like:
- fixing ShuffleNetV1 (Currently having an issue with ShuffleNetV1, where initializing the model fails).
- expose more PyTorch Lightning features.
- Better logging systems.
- Systematic testing.
- CI/CD with Travis.

### Prerequisites
- Python 3.6+
- PyTorch 1.7+
- lightning-bolts>=0.5.0
- pytorch-lightning>=1.6.2


### Training

```
# Start training with:
python main.py --model_arch [your model, e.g. `mobilenetv2`, `resnet18 resnet50 vgg16`, or `all` for all models]
```


### Accuracy

Models compared against the table reported in [the original repo](https://github.com/kuangliu/pytorch-cifar).
Some models were not reported in the original repo, and are represented with a `-`

| Model                                                 | [Orig Acc.](https://github.com/kuangliu/pytorch-cifar) | New Acc.[^1] |
|-------------------------------------------------------|--------------------------------------------------------|--------------|
| DenseNet-CIFAR10                                      | -                                                      | 93.44%       |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%                                                 | 94.94%       |
| [DenseNet161](https://arxiv.org/abs/1608.06993)       | -                                                      | -            |
| [DenseNet169](https://arxiv.org/abs/1608.06993)       | -                                                      | -            |
| [DenseNet201](https://arxiv.org/abs/1608.06993)       | -                                                      | -            |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%                                                 | 94.29%       |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%                                                 | 93.04%       |
| [DPN26](https://arxiv.org/abs/1707.01629)             | -                                                      | 95.03%       |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%                                                 | -            |
| [EfficientNetB0](https://arxiv.org/abs/1905.11946)    | -                                                      | 90.38%       |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)          | -                                                      | -            |
| [LeNet](yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | -                                                      | -            |
| [MobileNetV1](https://arxiv.org/abs/1704.04861)       | -                                                      | 91.30%       |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%                                                 | 93.17%       |
| [PNASNetA](https://arxiv.org/abs/1712.00559v3)        | -                                                      | -            |
| [PNASNetB](https://arxiv.org/abs/1712.00559v3)        | -                                                      | -            |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%                                                 | -            |
| [PreActResNet34](https://arxiv.org/abs/1603.05027)    | -                                                      | -            |
| [PreActResNet50](https://arxiv.org/abs/1603.05027)    | -                                                      | -            |
| [PreActResNet101](https://arxiv.org/abs/1603.05027)   | -                                                      | 94.33%       |
| [PreActResNet152](https://arxiv.org/abs/1603.05027)   | -                                                      | 94.83%       |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%                                                 | 94.27%       |
| [RegNetX_400MF](https://arxiv.org/abs/2003.13678)     | -                                                      | 94.61%       |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%                                                 | 94.61%       |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%                                                 | 94.30%       |
| [ResNet34](https://arxiv.org/abs/1512.03385)          | -                                                      | 93.86%       |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%                                                 | 94.73%       |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%                                                 | -            |
| [ResNet152](https://arxiv.org/abs/1512.03385)         | -                                                      | -            |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%                                                 | -            |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%                                                 | 95.18%       |
| [ResNeXt29(4x64d)](https://arxiv.org/abs/1611.05431)  | -                                                      | 95.67%       |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%                                                 | -            |


[^1]: Using default training config of `main.py` (i.e., 200 training epochs, initial learning rate of 0.05, etc), more configuration could give better results.
