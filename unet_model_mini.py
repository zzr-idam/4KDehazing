import torch.nn.functional as F
import torch
import torch.nn as nn 
from unet_parts import DoubleConv, Down, Up, OutConv


class UNet_mini(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet_mini, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.pre = nn.Conv2d(8, 3, 3, 1, 1)
        self.re = nn.Sigmoid()

        

    def forward(self, xs):
        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down4(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.re(self.pre(x))
        return x
