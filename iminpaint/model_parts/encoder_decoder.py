import torch
from torch import nn
from torch.nn import functional as F

from iminpaint.model_parts.gated_convolution import GatedConv

# * Mirror Padding
# * No batch normalization
# * ELU instead of ReLU
# * Contextual attention


class ContextualAttention(nn.Module):
    """ See Figure 3 in the paper for more information.
    """

    def __init__(self, use_attention_propagation=True, softmax_scale=10, rate=2, stride=1, ksize=3, fuse_k=3):
        super().__init__()
        self.use_attention_propagation = use_attention_propagation
        self.softmax_scale = softmax_scale
        self.rate = rate
        self.stride = stride
        self.ksize = ksize
        self.fuse_k = fuse_k

        self.register_parameter(
            'fuse_kernel',
            nn.Parameter(
                torch.eye(fuse_k).view(1, 1, fuse_k, fuse_k),
                requires_grad=False
            )
        )

    def forward(self, foreground, background, mask):
        rate, stride, ksize, fuse_k = \
            self.rate, self.stride, self.ksize, self.fuse_k
        padding = int(ksize // 2)

        # Original image to copy results to
        cols = self.img_2_col(background, 2 * rate, rate * stride)

        if rate != 1:
            foreground = F.interpolate(foreground, scale_factor=1. / rate)
            background = F.interpolate(background, scale_factor=1. / rate)
            mask = F.interpolate(mask, scale_factor=1. / rate)

        # Mask for background patches
        mask_cols = self.img_2_col(mask, ksize, stride)
        masked_cols = mask_cols.sum((2, 3, 4)) == 0.

        # Get the kernels from the background patch to convolve the
        # foreground with
        background_kernels = self.img_2_col(background, ksize, stride)

        batch_outs = []
        for i, (img, kernels, target) in enumerate(zip(foreground, background_kernels, cols)):
            # img: Shape (c, h, w)
            # kernels: Shape ((h // rate) * (w // rate), c, k, k)
            # target: Shape ((h // rate) * (w // rate), c, rate * 2, rate * 2)
            denom = torch.sqrt(torch.sum(kernels.pow(2), dim=(1, 2, 3), keepdim=True))
            kernels_normed = kernels / torch.max(denom, torch.empty_like(denom).fill_(1e-3))

            # Shape: (1, (h // rate) * (w // rate), (h // rate), (w // rate))
            out = F.conv2d(img.unsqueeze(0), kernels_normed, padding=padding)

            if self.use_attention_propagation:
                _, _, h, w = out.shape
                padding_fuse = int(fuse_k // 2)

                # Convolve output and transposed output with identity kernel
                out = out.reshape(1, 1, h * w, h * w)
                out = F.conv2d(out, self.fuse_kernel, padding=padding_fuse)
                out = (out.reshape(1, h, w, h, w)
                          .permute(0, 2, 1, 4, 3)
                          .reshape(1, 1, h * w, h * w))

                out = F.conv2d(out, self.fuse_kernel, padding=padding_fuse)
                out = (out.reshape(1, h, w, h, w)
                          .permute(0, 2, 1, 4, 3)
                          .reshape(1, h * w, h, w))

            out[:, masked_cols[i]] = -1000
            out = torch.softmax(self.softmax_scale * out, dim=1)

            out = F.conv_transpose2d(out, target, stride=rate, padding=padding)

            batch_outs.append(out)

        batch_outs = torch.cat(batch_outs, dim=0)
        return batch_outs

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
                GatedConv(128 * width, 128 * width),
                GatedConv(128 * width, 128 * width, activation=nn.ReLU()),
            )
            self.contextual_attention = ContextualAttention()
            self.attention_branch_cont = nn.Sequential(
                GatedConv(128 * width, 128 * width),
                GatedConv(128 * width, 128 * width),
            )
            decoder_inp_dim = int(128 * width) * 2
        else:
            decoder_inp_dim = int(128 * width)

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

    def forward(self, inp, mask=None):
        enc = self.encoder(inp)

        if self.use_contextual_attention:
            context = self.attention_branch(inp)
            mask_resized = F.interpolate(mask, size=context.size()[-2:])
            context = self.contextual_attention(context, context, mask_resized)
            context = self.attention_branch_cont(context)
            enc = torch.cat([enc, context], dim=1)

        return self.decoder(enc)


if __name__ == '__main__':
    net = EncoderDecoder(1, True)
    test_inp = torch.zeros(4, 5, 256, 256)
    output = net(test_inp, test_inp)
    assert output.shape == (4, 3, 256, 256)

    att = ContextualAttention(use_attention_propagation=True)
    output = att(test_inp, test_inp, test_inp)
    assert test_inp.shape == output.shape
