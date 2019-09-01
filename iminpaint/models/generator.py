from torch import nn


class Generator(nn.Module):
    def __init__(self, feature_map_scale=8, latent_vector_size=100, n_gpu=1):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_vector_size, feature_map_scale * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_scale * 8),
            nn.LeakyReLU(0.1, True),
            # state size. (feature_map_scale*8) x 4 x 4
            nn.ConvTranspose2d(feature_map_scale * 8, feature_map_scale * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_scale * 4),
            nn.LeakyReLU(0.1, True),
            # state size. (feature_map_scale*4) x 8 x 8
            nn.ConvTranspose2d(feature_map_scale * 4, feature_map_scale * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_scale * 2),
            nn.LeakyReLU(0.1, True),
            # state size. (feature_map_scale*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_scale * 2, feature_map_scale, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_scale),
            nn.LeakyReLU(0.1, True),
            # state size. (feature_map_scale) x 32 x 32
            nn.ConvTranspose2d(feature_map_scale, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, inp):
        return self.main(inp)

