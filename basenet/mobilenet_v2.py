from collections import namedtuple

import torch
from torchvision import models
from torchvision.models.mobilenet import model_urls
from torchutil import *
import os

weights_folder = os.path.join(os.path.dirname(__file__) + '/../pretrain')


class mobilenet_v2(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super(mobilenet_v2, self).__init__()
        model_urls['mobilenet_v2'] = model_urls['mobilenet_v2'].replace('https://', 'http://')
        mobilenet_features = models.mobilenet_v2(pretrained=False)
        if pretrained:
            mobilenet_features.load_state_dict(
                copyStateDict(torch.load(os.path.join(weights_folder, 'mobilenet_v2-b0353104.pth'))))
        mobilenet_features = mobilenet_features.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):  # conv2_2
            self.slice1.add_module(str(x), mobilenet_features[x])
        for x in range(2, 7):  # conv3_3
            self.slice2.add_module(str(x), mobilenet_features[x])
        for x in range(7, 14):  # conv4_3
            self.slice3.add_module(str(x), mobilenet_features[x])
        for x in range(14, 18):  # conv5_3
            self.slice4.add_module(str(x), mobilenet_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(320, 640, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(640, 640, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
