# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:59
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : model.py
"""

"""
import torch.nn as nn
import torch
from .unet_base import *
from .nested_unet_base import *


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.bilinear = cfg.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, self.bilinear)
        self.up2 = Up(512, 128, self.bilinear)
        self.up3 = Up(256, 64, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class NestedUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.deepsupervision = cfg.deepsupervision
        self.bilinear = cfg.bilinear

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(
            nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(
            nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(
            nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(
            nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(
            nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(
            nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.sigmoid = nn.Sigmoid()

        if self.deepsupervision:
            self.final1 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class NestedUNetPlus(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.deepsupervision = cfg.deepsupervision
        self.bilinear = cfg.bilinear

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1_0 = VGGBlock(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_1_1 = VGGBlock(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_1_2 = VGGBlock(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv1_1 = VGGBlock(
            nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(
            nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(
            nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2_0 = VGGBlock(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2_1 = VGGBlock(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2_2 = VGGBlock(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv1_2 = VGGBlock(
            nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(
            nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3_2 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3_1 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3_0 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv1_3 = VGGBlock(
            nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4_2 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4_1 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4_0 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.sigmoid = nn.Sigmoid()

        if self.deepsupervision:
            self.final1_0 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final2_0 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final3_0 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final4_0 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)

            self.final1_1 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final2_1 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final3_1 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final4_1 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)

            self.final1_2 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final2_2 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final3_2 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final4_2 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))

        x0_1_2 = self.conv0_1_2(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1_1 = self.conv0_1_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1_0 = self.conv0_1_0(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))

        x0_2_2 = self.conv0_2_2(torch.cat([x0_0, x0_1_2, self.up(x1_1)], 1))
        x0_2_1 = self.conv0_2_1(torch.cat([x0_0, x0_1_1, self.up(x1_1)], 1))
        x0_2_0 = self.conv0_2_0(torch.cat([x0_0, x0_1_0, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        x0_3_2 = self.conv0_3_2(torch.cat([x0_0, x0_1_2, x0_2_2, self.up(x1_2)], 1))
        x0_3_1 = self.conv0_3_1(torch.cat([x0_0, x0_1_1, x0_2_1, self.up(x1_2)], 1))
        x0_3_0 = self.conv0_3_0(torch.cat([x0_0, x0_1_0, x0_2_0, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4_2 = self.conv0_4_2(torch.cat([x0_0, x0_1_2, x0_2_2, x0_3_2, self.up(x1_3)], 1))
        x0_4_1 = self.conv0_4_1(torch.cat([x0_0, x0_1_1, x0_2_1, x0_3_1, self.up(x1_3)], 1))
        x0_4_0 = self.conv0_4_0(torch.cat([x0_0, x0_1_0, x0_2_0, x0_3_0, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1_2 = self.final1_2(x0_1_2)
            output2_2 = self.final2_2(x0_2_2)
            output3_2 = self.final3_2(x0_3_2)
            output4_2 = self.final4_2(x0_4_2)

            output1_1 = self.final1_1(x0_1_1)
            output2_1 = self.final2_1(x0_2_1)
            output3_1 = self.final3_1(x0_3_1)
            output4_1 = self.final4_1(x0_4_1)

            output1_0 = self.final1_0(x0_1_0)
            output2_0 = self.final2_0(x0_2_0)
            output3_0 = self.final3_0(x0_3_0)
            output4_0 = self.final4_0(x0_4_0)

            output_1 = torch.stack((output1_0, output1_1, output1_2), dim=1).squeeze(2)
            output_2 = torch.stack((output2_0, output2_1, output2_2), dim=1).squeeze(2)
            output_3 = torch.stack((output3_0, output3_1, output3_2), dim=1).squeeze(2)
            output_4 = torch.stack((output4_0, output4_1, output4_2), dim=2).squeeze(2)

            return [output_1, output_2, output_3, output_4]

        else:
            output = self.final(x0_4)
            return output

# 0 b, 1 l, 2 s

class NestedUNetPlusAdd(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.deepsupervision = cfg.deepsupervision
        self.bilinear = cfg.bilinear

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1_0 = VGGBlock(
            2*nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_1_1 = VGGBlock(
            2*nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_1_2 = VGGBlock(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv1_1 = VGGBlock(
            nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(
            nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(
            nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2_0 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2_1 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2_2 = VGGBlock(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv1_2 = VGGBlock(
            nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(
            nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3_2 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3_1 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3_0 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv1_3 = VGGBlock(
            nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4_2 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4_1 = VGGBlock(
            nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4_0 = VGGBlock(
            nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0])

        self.sigmoid = nn.Sigmoid()

        if self.deepsupervision:
            self.final1_0 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final2_0 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final3_0 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final4_0 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)

            self.final1_1 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final2_1 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final3_1 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final4_1 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)

            self.final1_2 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final2_2 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final3_2 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
            self.final4_2 = nn.Conv2d(
                nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))

        x0_1_2 = self.conv0_1_2(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1_1 = self.conv0_1_1(torch.cat([x0_0, self.up(x1_0), x0_1_2], 1))
        x0_1_0 = self.conv0_1_0(torch.cat([x0_0, self.up(x1_0), x0_1_1], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))

        x0_2_2 = self.conv0_2_2(torch.cat([x0_0, x0_1_2, self.up(x1_1)], 1))
        x0_2_1 = self.conv0_2_1(torch.cat([x0_0, x0_1_1, self.up(x1_1), x0_2_2], 1))
        x0_2_0 = self.conv0_2_0(torch.cat([x0_0, x0_1_0, self.up(x1_1), x0_2_1], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        x0_3_2 = self.conv0_3_2(torch.cat([x0_0, x0_1_2, x0_2_2, self.up(x1_2)], 1))
        x0_3_1 = self.conv0_3_1(torch.cat([x0_0, x0_1_1, x0_2_1, self.up(x1_2), x0_3_2], 1))
        x0_3_0 = self.conv0_3_0(torch.cat([x0_0, x0_1_0, x0_2_0, self.up(x1_2), x0_3_1], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4_2 = self.conv0_4_2(torch.cat([x0_0, x0_1_2, x0_2_2, x0_3_2, self.up(x1_3)], 1))
        x0_4_1 = self.conv0_4_1(torch.cat([x0_0, x0_1_1, x0_2_1, x0_3_1, self.up(x1_3), x0_4_2], 1))
        x0_4_0 = self.conv0_4_0(torch.cat([x0_0, x0_1_0, x0_2_0, x0_3_0, self.up(x1_3), x0_4_1], 1))

        if self.deepsupervision:
            output1_2 = self.final1_2(x0_1_2)
            output2_2 = self.final2_2(x0_2_2)
            output3_2 = self.final3_2(x0_3_2)
            output4_2 = self.final4_2(x0_4_2)

            output1_1 = self.final1_1(x0_1_1)
            output2_1 = self.final2_1(x0_2_1)
            output3_1 = self.final3_1(x0_3_1)
            output4_1 = self.final4_1(x0_4_1)

            output1_0 = self.final1_0(x0_1_0)
            output2_0 = self.final2_0(x0_2_0)
            output3_0 = self.final3_0(x0_3_0)
            output4_0 = self.final4_0(x0_4_0)

            output_1 = torch.stack((output1_0, output1_1, output1_2), dim=1).squeeze(2)
            output_2 = torch.stack((output2_0, output2_1, output2_2), dim=1).squeeze(2)
            output_3 = torch.stack((output3_0, output3_1, output3_2), dim=1).squeeze(2)
            output_4 = torch.stack((output4_0, output4_1, output4_2), dim=2).squeeze(2)

            return [output_1, output_2, output_3, output_4]

        else:
            output = self.final(x0_4)
            return output