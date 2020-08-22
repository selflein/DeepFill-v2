import torch
from torch import nn

from iminpaint.model_parts.encoder_decoder import EncoderDecoder


class Generator(nn.Module):
    """
    Two stage:
      1. Encoder-Decoder net trained with spatially-discounted L1 reconstruction
       loss (weighting gamma^l where l is distance to next know pixel and
       gamma=0.99)
      2. Refinement encoder-decoder net trained with L1 and GAN loss
    """

    def __init__(self, width=1, use_contextual_attention=True):
        super().__init__()
        self.coarse_encoder_decoder = EncoderDecoder(
            width=width, use_contextual_attention=False)

        self.fine_encoder_decoder = EncoderDecoder(
            width=width, use_contextual_attention=use_contextual_attention)

    def forward(self, masked_img, mask, sketch=None):
        if sketch is None:
            sketch = torch.zeros_like(mask)

        inp = torch.cat([masked_img, mask, sketch], dim=1)
        coarse_result = self.coarse_encoder_decoder(inp)

        # Paste coarse result into the image at mask
        pasted_coarse_result = masked_img + (1 - mask) * coarse_result
        refinement_inp = torch.cat([pasted_coarse_result, mask, sketch], dim=1)

        fine_result = self.fine_encoder_decoder(refinement_inp)
        return fine_result, coarse_result


if __name__ == '__main__':
    net = Generator(use_contextual_attention=False)
    out_coarse, out_fine = net(torch.zeros(1, 3, 256, 256),
                               torch.zeros(1, 1, 256, 256),
                               torch.zeros(1, 1, 256, 256))

    assert out_coarse.shape == (1, 3, 256, 256)
    assert out_fine.shape == (1, 3, 256, 256)
