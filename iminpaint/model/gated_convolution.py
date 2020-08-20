from torch import nn


class GatedConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.feature_branch = nn.Sequential(
            nn.Conv2d(*args, **kwargs),
            nn.LeakyReLU()
        )
        self.gating_branch = nn.Sequential(
            nn.Conv2d(*args, **kwargs),
            nn.Sigmoid()
        )

    def forward(self, inp):
        features = self.feature_branch(inp)
        gates = self.gating_branch(inp)

        return features * gates
