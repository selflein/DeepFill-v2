import torch
from torch import nn
from torch.nn import functional as F


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args,
                 padding_mode='replicate', kernel_size=3, stride=1, dilation=1,
                 activation=F.elu, **kwargs):
        super().__init__()
        padding = int(dilation * (kernel_size - 1) / 2)
        self.activation = activation
        self.out_channels = int(out_channels)

        self.conv = nn.Conv2d(int(in_channels), int(out_channels * 2), *args,
                              **kwargs,
                              padding_mode=padding_mode,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

    def forward(self, inp):
        out = self.conv(inp)
        features, gates = torch.split(out, self.out_channels, dim=1)
        features = self.activation(features)
        gates = F.sigmoid(gates)

        return features * gates
