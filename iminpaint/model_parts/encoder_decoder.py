import torch
from torch import nn
from torch.nn import functional as F

from iminpaint.model_parts.gated_convolution import GatedConv

# * Mirror Padding
# * No batch normalization
# * ELU instead of ReLU
# * Contextual attention


class ContextualAttention(nn.Module):

    def __init__(self, use_attention_propagation=False, softmax_scale=10):
        super().__init__()
        self.use_attention_propagation = use_attention_propagation
        self.softmax_scale = 10

    # TODO: Implement contextual attention
    def forward(self, foreground, background, mask, rate=2, stride=1, ksize=3):
        # Original image size to copy results to
        background_cols = self.img_2_col(background, 2 * rate, rate * stride)

        if rate != 1:
            foreground = F.upsample(foreground, scale_factor=1. / rate)
            background = F.upsample(background, scale_factor=1. / rate)
            mask = F.upsample(mask, scale_factor=1. / rate)

        mask_cols = self.img_2_col(mask, ksize, stride)
        masked_cols = (mask_cols.sum(2, 3, 4) == 0.).float()

        background_kernels = self.img_2_col(background, ksize, stride)

        for img, kernels, target in zip(foreground, background_kernels, background_cols):
            kernels_normed = kernels / max(torch.sqrt(torch.sum(kernels.pow(2), dim=(1, 2, 3))), 1e-4)
            out = F.conv2d(img.unsqueeze(0), kernels_normed, padding=1)

            if self.use_attention_propagation:
                # TODO Convolve output and transposed output with identity kernel
                pass

            out[mask_cols]
            out = torch.softmax(self.softmax_scale * out, dim=1)

        return inp

    @staticmethod
    def img_2_col(img, ksize, stride):
        b, c, h, w = img.shape
        # Extract `o` conv kernels based on feature map
        # (b, c, h, w) -> (b, c * k * k, o)
        conv_kernels = F.unfold(
            img,
            kernel_size=(ksize, ksize),
            dilation=(1, 1),
            padding=(1, 1),
            stride=(stride, stride)
        )

        # (b, c * k * k, o) -> (b, o, c, k, k)
        conv_kernels = (conv_kernels.reshape(b, c, ksize, ksize, -1)
                                    .permute(0, 4, 1, 2, 3))
        return conv_kernels


class EncoderDecoder(nn.Module):

    def __init__(self, width=1, use_contextual_attention=False):
        super().__init__()
        self.use_contextual_attention = use_contextual_attention

        self.encoder = nn.Sequential(
            GatedConv(5, 32 * width, kernel_size=5, stride=1),
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
            self.attention_branch = nn.Sequential(
                GatedConv(5, 32 * width, kernel_size=5, stride=1),
                GatedConv(32 * width, 32 * width, stride=2),
                GatedConv(32 * width, 64 * width),
                GatedConv(64 * width, 128 * width, stride=2),
                GatedConv(64 * width, 128 * width),
                GatedConv(64 * width, 128 * width, activation=nn.ReLU()),
            )
            self.contextual_attention = ContextualAttention()
            self.attention_branch_cont = nn.Sequential(
                GatedConv(128 * width, 128 * width),
                GatedConv(128 * width, 128 * width),
            )
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
            GatedConv(16 * width, 3, activation=None),
            nn.Sigmoid()
        )

    def forward(self, inp):
        enc = self.encoder(inp)

        if self.use_contextual_attention:
            context = self.attention_branch(inp)
            context = self.contextual_attention(context)
            context = self.attention_branch_cont(context)
            enc = torch.cat([enc, context], dim=1)

        return self.decoder(enc)


if __name__ == '__main__':
    net = EncoderDecoder(1, False)
    test_inp = torch.zeros(4, 5, 256, 256)
    output = net(test_inp)
    assert output.shape == (4, 3, 256, 256)
