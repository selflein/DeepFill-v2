import torch
from torch import nn


class SpecConv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.spec_cnn = nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inp):
        out = self.spec_cnn(inp)
        out = self.leaky_relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, c_base=64):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            SpecConv2d(5, c_base, kernel_size=5, stride=1, padding=2),
            SpecConv2d(c_base, c_base * 2, kernel_size=5, stride=2, padding=2),
            SpecConv2d(c_base * 2, c_base * 4, kernel_size=5, stride=2, padding=2),
            SpecConv2d(c_base * 4, c_base * 4, kernel_size=5, stride=2, padding=2),
            SpecConv2d(c_base * 4, c_base * 4, kernel_size=5, stride=2, padding=2),
            nn.utils.spectral_norm(
                nn.Conv2d(c_base * 4, c_base * 4, kernel_size=5, stride=2, padding=2)
            )
        )

    def forward(self, img, mask, sketch=None):
        if sketch is None:
            sketch = torch.zeros_like(mask)

        inp = torch.cat([img, mask, sketch], dim=1)
        return self.main(inp)


if __name__ == "__main__":
    test_net = Discriminator(64)
    output = test_net(torch.zeros(4, 3, 256, 256),
                      torch.zeros(4, 1, 256, 256),
                      torch.zeros(4, 1, 256, 256))
    assert output.shape == (4, 256, 8, 8)
