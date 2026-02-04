"""
 this file can be used independently inside VCSL without pulling
specific utilities.
"""
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Conv(nn.Module):
    """Minimal Conv block: Conv2d -> BN -> Activation (SiLU by default)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, *, padding: Optional[int] = None,
                 stride: int = 1, activation: Optional[str] = "SiLU",groups:int=1):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2)
        if activation == "SiLU":
            self.act = nn.SiLU()
        elif activation is None or activation is False:
            self.act = nn.Identity()
        else:
            # fallback to SiLU for unknown names
            self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DConv(nn.Module):
    """
    DConv adapted from YOLO `module.py`.

    Usage: given a feature map r (B, C, H, W), returns a refined tensor of same shape.
    """

    def __init__(self, in_channels: int = 512, alpha: float = 0.8, atoms: int = 512):
        super().__init__()
        self.alpha = alpha

        # channel generation -> group-wise interaction -> project back
        self.CG = Conv(in_channels, atoms, kernel_size=1)
        # depthwise conv (groups=atoms)
        self.GIE = Conv(atoms, atoms, kernel_size=5, padding=2, activation=False,groups=atoms)
        self.D = Conv(atoms, in_channels, kernel_size=1, activation=False)

        try:
            with torch.no_grad():
                self.D.conv.weight.zero_()
        except Exception:

            pass

    def PONO(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-5)
        return x

    def forward(self, r: Tensor) -> Tensor:
        x = self.CG(r)
        x = self.GIE(x)
        x = self.PONO(x)
        x = self.D(x)
        return  self.alpha * x


__all__ = ["DConv"]
