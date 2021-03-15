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


class AI85FaceIDNetResidual(nn.Module):
    """
    FaceNet Model With Residual Connections
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
                                          bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(16, 32, 3, padding=1, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedConv2dReLU(32, 64, 3, padding=1, bias=bias, **kwargs)
        
        self.maxpool1 = ai8x.MaxPool2d(2, 2)
        self.conv4 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        
        self.maxpool2 = ai8x.MaxPool2d(2, 2)
        self.conv6 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.conv7 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        
        self.maxpool3 = ai8x.MaxPool2d(2, 2)
        self.conv8 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.conv9 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        
        self.maxpool4 = ai8x.MaxPool2d(2, 2)
        self.conv10 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.conv11 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        
        self.conv12 = ai8x.FusedMaxPoolConv2d(64, 512, 1, pool_size=2, pool_stride=2,
                                             padding=0, bias=bias, **kwargs)
        self.avgpool = ai8x.AvgPool2d((3, 5))

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.maxpool1(x)
        res = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + res
        
        x = self.maxpool2(x)
        res = x
        x = self.conv6(x)
        x = self.conv7(x)
        x = x + res
        
        x = self.maxpool3(x)
        res = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = x + res
        
        x = self.maxpool4(x)
        res = x
        x = self.conv10(x)
        x = self.conv11(x)
        x = x + res
        
        x = self.conv12(x)
        x = self.avgpool(x)
        return x


def ai85faceidnetres(pretrained=False, **kwargs):
    """
    Constructs a FaceIDNet model.
    """
    assert not pretrained
    return AI85FaceIDNetResidual(**kwargs)


models = [
    {
        'name': 'ai85faceidnetres',
        'min_input': 1,
        'dim': 3,
    },
]
