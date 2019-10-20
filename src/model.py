import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import sys
sys.path.append("../input/pretrainedmodels")
from senet import se_resnext101_32x4d, se_resnext50_32x4d, senet154
from inceptionresnetv2 import inceptionresnetv2
from efficientnet_pytorch import EfficientNet

import sys
sys.path.append("../src/")
from layer import AdaptiveConcatPool2d, Flatten, SEBlock, GeM


encoders = {
    "se_resnext50_32x4d": {
        "encoder": se_resnext50_32x4d,
        "out_shape": 2048
    },
    "se_resnext101_32x4d": {
        "encoder": se_resnext101_32x4d,
        "out_shape": 2048
    },
    "inceptionresnetv2": {
        "encoder": inceptionresnetv2,
        "out_shape": 1536
    },
    "resnet34": {
        "encoder": models.resnet34,
        "out_shape": 512
    },
    "resnet50": {
        "encoder": models.resnet50,
        "out_shape": 2048
    }
}

def create_net(net_cls, pretrained: bool):
    net = net_cls()
    if pretrained is not None:
        net.load_state_dict(torch.load(pretrained))
    #net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=models.resnet50):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AdaptiveConcatPool2d()
        self.net.fc = nn.Sequential(
            Flatten(),
            SEBlock(2048*2),
            nn.Dropout(),
            nn.Linear(2048*2, num_classes)
        )

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=models.densenet121):
        super().__init__()
        self.net = net_cls(pretrained=pretrained)
        self.avg_pool = AdaptiveConcatPool2d()
        self.net.classifier = nn.Sequential(
            Flatten(),
            SEBlock(1024*2),
            nn.Dropout(),
            nn.Linear(1024*2, num_classes)
        )

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out)
        out = self.net.classifier(out)
        return out


class CnnModel(nn.Module):
    def __init__(self, num_classes, encoder="se_resnext50_32x4d", pretrained="imagenet", pool_type="concat"):
        super().__init__()
        self.net = encoders[encoder]["encoder"](pretrained=pretrained)

        if encoder in ["resnet34", "resnet50"]:
            if pool_type == "concat":
                self.net.avgpool = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avgpool = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.fc = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )
        elif encoder == "inceptionresnetv2":
            if pool_type == "concat":
                self.net.avgpool_1a = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avgpool_1a = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.last_linear = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )
        else:
            if pool_type == "concat":
                self.net.avg_pool = AdaptiveConcatPool2d()
                out_shape = encoders[encoder]["out_shape"] * 2
            elif pool_type == "avg":
                self.net.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                out_shape = encoders[encoder]["out_shape"]
            elif pool_type == "gem":
                self.net.avg_pool = GeM()
                out_shape = encoders[encoder]["out_shape"]
            self.net.last_linear = nn.Sequential(
                Flatten(),
                SEBlock(out_shape),
                nn.Dropout(),
                nn.Linear(out_shape, num_classes)
            )


    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class Efficient(nn.Module):
    def __init__(self, num_classes, encoder='efficientnet-b0', pool_type="avg"):
        super().__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        self.net = EfficientNet.from_pretrained(encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool_type == "concat":
            self.net.avg_pool = AdaptiveConcatPool2d()
            out_shape = n_channels_dict[encoder]*2
        elif pool_type == "avg":
            self.net.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            out_shape = n_channels_dict[encoder]
        elif pool_type == "gem":
            self.net.avg_pool = GeM()
            out_shape = n_channels_dict[encoder]
        self.classifier = nn.Sequential(
            Flatten(),
            SEBlock(out_shape),
            nn.Dropout(),
            nn.Linear(out_shape, num_classes)
        )

    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.avg_pool(x)
        x = self.classifier(x)

        return x


class InceptionResNetV2_V2(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet"):
        super().__init__()
        self.net = inceptionresnetv2(pretrained=pretrained)
        self.net.avgpool_1a = AdaptiveConcatPool2d()
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(1536*2),
            nn.Dropout(),
            nn.Linear(1536*2, num_classes)
        )


    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SENetV4(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet"):
        super().__init__()
        self.net = senet154(pretrained=pretrained)
        self.net.avg_pool = AdaptiveConcatPool2d()
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(2048*2),
            nn.Dropout(),
            nn.Linear(2048*2, num_classes)
        )

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class PNASNet5Large(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet"):
        super().__init__()
        self.net = pnasnet5large(pretrained=pretrained)
        self.net.avg_pool = AdaptiveConcatPool2d()
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(4320 * 2),
            nn.Dropout(),
            nn.Linear(4320 * 2, num_classes)
        )

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class Xception(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet"):
        super().__init__()
        self.net = xception(pretrained=pretrained)
        self.avg_pool = AdaptiveConcatPool2d()
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(2048 * 2),
            nn.Dropout(),
            nn.Linear(2048 * 2, num_classes)
        )

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out)
        out = self.net.last_linear(out)
        return out
