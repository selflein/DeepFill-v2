from torch import nn


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args,
                 padding_mode='replicate', kernel_size=3, stride=1, dilation=1,
                 **kwargs):
        super().__init__()
        padding = int(dilation * (kernel_size - 1) / 2)

        self.feature_branch = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), *args, **kwargs,
                      padding_mode=padding_mode, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.ELU()
        )
        self.gating_branch = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), *args, **kwargs,
                      padding_mode=padding_mode, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.Sigmoid()
        )

    def forward(self, inp):
        features = self.feature_branch(inp)
        gates = self.gating_branch(inp)

        return features * gates
