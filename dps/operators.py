"""Differentiable Gaussian measurement operator."""

import torch
from torch import nn
from torch.nn import functional as F


def gaussian_kernel2d(ksize=61, sigma=3.0):
    if ksize < 1 or ksize % 2 != 1 or sigma <= 0:
        raise ValueError("Use a positive odd kernel size and positive sigma.")
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    kernel = torch.outer(gauss, gauss)
    return kernel / kernel.sum()


class GaussianBlur(nn.Module):
    """Channelwise Gaussian convolution with reflection padding."""

    def __init__(self, ksize=61, sigma=4.0):
        super().__init__()
        self.ksize = ksize
        self.register_buffer("kernel", gaussian_kernel2d(ksize, sigma))

    def forward(self, x):
        channels = x.shape[1]
        kernel = self.kernel.expand(channels, 1, self.ksize, self.ksize)
        pad = self.ksize // 2
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"),
                        kernel, groups=channels)
