# PyTorch Lightning CIFAR10

## About this fork

Modified version of the [PyTorch CIFAR project](https://github.com/kuangliu/pytorch-cifar) to exploit the [PyTorch Lightning package](https://www.pytorchlightning.ai/).

In addition:
- Improvements `main.py` script, allowing you to train one or more models in a single command
- Default optimizer changed to Adam
- [`black`](https://github.com/psf/black) formatter applied to all files.
- Added more consistnet config of VGG and ShuffleNetV2 models

Currently supported models:

- [ ] densenet.py
- [ ] dla.py
- [ ] dla_simple.py
- [ ] dpn.py
- [ ] efficientnet.py
- [ ] googlenet.py
- [ ] lenet.py
- [x] mobilenet.py
- [x] mobilenetv2.py
- [ ] pnasnet.py
- [ ] preact_resnet.py
- [ ] regnet.py
- [ ] resnet.py
- [ ] resnext.py
- [ ] senet.py
- [ ] shufflenet.py
- [ ] shufflenetv2.py
- [ ] vgg.py

## Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

### Prerequisites
- Python 3.6+
- PyTorch 1.0+

### Training
```
# Start training with: 
python main.py --model_name [your model, e.g. `mobilenetv2`, or `all` for all models]
```

### Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

