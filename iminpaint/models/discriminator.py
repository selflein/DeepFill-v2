from torch import nn


class Discriminator(nn.Module):
    def __init__(self, inp_channels=3, feature_map_scale=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (inp_channels) x 64 x 64
            nn.Conv2d(inp_channels, feature_map_scale, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_map_scale) x 32 x 32
            nn.Conv2d(feature_map_scale, feature_map_scale * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_scale * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_map_scale*2) x 16 x 16
            nn.Conv2d(feature_map_scale * 2, feature_map_scale * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_scale * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_map_scale*4) x 8 x 8
            nn.Conv2d(feature_map_scale * 4, feature_map_scale * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_scale * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_map_scale*8) x 4 x 4
            nn.Conv2d(feature_map_scale * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        return self.main(inp)
