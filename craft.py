"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.init as init
from torchutil import *

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, activation_layer=nn.ReLU):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            activation_layer(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            activation_layer(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=True, freeze=False, backbone='mobilenet'):
        super(CRAFT, self).__init__()
        
        init_func = {
            'vgg': self.init_vgg,
            'mobilenet': self.init_mobilenet,
        }
        assert backbone in init_func, "This backbone architecture is not supported!"
        out_channels = init_func[backbone](pretrained, freeze)
        
        num_class = 15
        
        # self.conv_cls = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels // 2, num_class, kernel_size=1),
        # )
        
        self.conv_cls = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
        
    def init_vgg(self, pretrained, freeze):
        from basenet.vgg16_bn import vgg16_bn
        
        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)
        
        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)
        
        return 32
        
    def init_mobilenet(self, pretrained, freeze):
        from basenet.mobilenet_v2 import mobilenet_v2
    
        """ Base network """
        self.basenet = mobilenet_v2(pretrained, freeze)
        
        """ U network """
        self.upconv1 = double_conv(640, 320, 64, nn.ReLU6)
        self.upconv2 = double_conv(96, 64, 32, nn.ReLU6)
        self.upconv3 = double_conv(32, 32, 16, nn.ReLU6)
        self.upconv4 = double_conv(16, 16, 16, nn.ReLU6)
        
        return 16
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)
        
        # for x in sources:
        #     print(x.shape)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature

import time
if __name__ == '__main__':
    model = CRAFT(pretrained=True, backbone='mobilenet').cuda(0)
    print("Loaded successfully")
    t = time.time()
    output, _ = model(torch.randn(1, 3, 720, 1280).cuda(0))
    print(output.shape)
    print(time.time() - t, 'seconds')
