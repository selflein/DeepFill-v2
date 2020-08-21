import torch
from torch import nn

from iminpaint.model.gated_convolution import GatedConv

# * Mirror Padding
# * No batch normalization
# * ELU instead of ReLU
# * Contextual attention


class ContextualAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        # TODO: Implement contextual attention
        return inp


class EncoderDecoder(nn.Module):

    def __init__(self, width=1, use_contextual_attention=False):
        super().__init__()
        self.use_contextual_attention = use_contextual_attention

        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32 * width, kernel_size=5, padding=2,
                      padding_mode='replicate', stride=1),
            GatedConv(32 * width, 64 * width, stride=2),
            GatedConv(64 * width, 64 * width),
            GatedConv(64 * width, 128 * width, stride=2),
            GatedConv(128 * width, 128 * width),
            GatedConv(128 * width, 128 * width),
            GatedConv(128 * width, 128 * width, dilation=2),
            GatedConv(128 * width, 128 * width, dilation=4),
            GatedConv(128 * width, 128 * width, dilation=8),
            GatedConv(128 * width, 128 * width, dilation=16),
        )

        if self.use_contextual_attention:
            self.contextual_attention = ContextualAttention()
            decoder_inp_dim = int(128 * width) * 2
        else:
            decoder_inp_dim = 128 * width

        self.decoder = nn.Sequential(
            GatedConv(decoder_inp_dim, 128 * width),
            GatedConv(128 * width, 128 * width),
            nn.Upsample(scale_factor=2, mode='nearest'),
            GatedConv(128 * width, 64 * width),
            GatedConv(64 * width, 64 * width),
            nn.Upsample(scale_factor=2, mode='nearest'),
            GatedConv(64 * width, 32 * width),
            GatedConv(32 * width, 16 * width),
            nn.Conv2d(16 * width, 3, kernel_size=3, stride=1, padding=1,
                      padding_mode='replicate')
        )

    def forward(self, inp):
        enc = self.encoder(inp)

        if self.use_contextual_attention:
            context = self.contextual_attention(inp)
            enc = torch.cat([enc, context], dim=1)

        return self.decoder(enc)


if __name__ == '__main__':
    net = EncoderDecoder(1, False)
    test_inp = torch.zeros(4, 5, 256, 256)
    output = net(test_inp)
    assert output.shape == (4, 3, 256, 256)
