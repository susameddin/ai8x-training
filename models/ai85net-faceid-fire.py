###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
FaceID network for AI85/AI86

Optionally quantize/clamp activations
"""
import torch.nn as nn

import ai8x

import torch

class AI85FaceIDNetFire(nn.Module):
    """
    FaceNet Model With Fire Modules
    """
    def __init__(
            self,
            num_classes=None,  # pylint: disable=unused-argument
            num_channels=3,
            dimensions=(160, 120),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, padding=1,
                                          bias=False, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, 16, 1, pool_size=2, pool_stride=2,
                                          bias=False, **kwargs)
        self.conv3 = ai8x.FusedConv2dReLU(16, 64, 1,
                                          bias=False, **kwargs)
        self.conv4 = ai8x.FusedConv2dReLU(16, 64, 3, padding=1,
                                          bias=False, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(128, 32, 1, pool_size=2, pool_stride=2,
                                          bias=False, **kwargs)
        self.conv6 = ai8x.FusedConv2dReLU(32, 128, 1,
                                          bias=False, **kwargs)
        self.conv7 = ai8x.FusedConv2dReLU(32, 128, 3, padding=1,
                                          bias=False, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2dReLU(256, 16, 1, pool_size=2, pool_stride=2,
                                          bias=False, **kwargs)
        self.conv9 = ai8x.FusedConv2dReLU(16, 64, 1,
                                          bias=False, **kwargs)
        self.conv10 = ai8x.FusedConv2dReLU(16, 64, 3, padding=1,
                                          bias=False, **kwargs)
        self.conv11 = ai8x.FusedMaxPoolConv2dReLU(128, 32, 1, pool_size=2, pool_stride=2,
                                          bias=False, **kwargs)
        self.conv12 = ai8x.FusedConv2dReLU(32, 128, 1,
                                          bias=False, **kwargs)
        self.conv13 = ai8x.FusedConv2dReLU(32, 128, 3, padding=1,
                                          bias=False, **kwargs)
        self.conv14 = ai8x.FusedMaxPoolConv2d(256, 512, 1, pool_size=2, pool_stride=2,
                                             padding=0, bias=False, **kwargs)
        self.avgpool = ai8x.AvgPool2d((3, 5))

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        left = self.conv3(x)
        right = self.conv4(x)
        x = torch.cat([left, right], 1)
        x = self.conv5(x)
        left = self.conv6(x)
        right = self.conv7(x)
        x = torch.cat([left, right], 1)
        x = self.conv8(x)
        left = self.conv9(x)
        right = self.conv10(x)
        x = torch.cat([left, right], 1)
        x = self.conv11(x)
        left = self.conv12(x)
        right = self.conv13(x)
        x = torch.cat([left, right], 1)
        x = self.conv14(x)
        x = self.avgpool(x)
        return x


def ai85faceidnetfire(pretrained=False, **kwargs):
    """
    Constructs a FaceIDNet model.
    """
    assert not pretrained
    return AI85FaceIDNetFire(**kwargs)


models = [
    {
        'name': 'ai85faceidnetfire',
        'min_input': 1,
        'dim': 3,
    },
]