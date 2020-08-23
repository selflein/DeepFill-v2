from pathlib import Path

import torch
from torch import optim
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision.utils import make_grid
from hydra.utils import to_absolute_path

from iminpaint.data.dataloader import dataloaders
from iminpaint.model_parts.generator import Generator
from iminpaint.model_parts.discriminator import Discriminator


class DeepFill(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.gen = Generator(
            width=self.hparams.model.generator_width,
            use_contextual_attention=self.hparams.model.use_contextual_attention
        )
        self.disc = Discriminator(c_base=self.hparams.model.disc_c_base)

        self.train_loader, self.val_loader = dataloaders.create_train_val_loader(
            path=Path(to_absolute_path(self.hparams.data.path)),
            edges_path=Path(to_absolute_path(self.hparams.data.edges_path)),
            batch_size=self.hparams.data.batch_size,
            num_workers=self.hparams.data.num_workers
        )

    def forward(self, masked_img, mask, edges_mask):
        fine, _ = self.gen(masked_img, mask, edges_mask)
        return self.get_completed_img(masked_img, mask, fine)

    def training_step(self, batch, batch_idx, optimizer_idx) -> pl.TrainResult:
        # Update discriminator
        if optimizer_idx == 0:
            loss = self.discriminator_step(batch)
            res = pl.TrainResult(loss)
            res.log('train/disc_loss', loss, prog_bar=True)
            return res

        # Update generator
        if optimizer_idx == 1:
            if not batch_idx % 5 == 0:
                # Skip the optimization of generator for this batch
                # This is somewhat of a hack since PL does not easily allow
                # skipping a training step
                return {'loss': torch.tensor(0., requires_grad=True)}
            gen_loss, l1_loss = self.generator_step(batch)
            res = pl.TrainResult(gen_loss + l1_loss)
            res.log('train/l1_loss', l1_loss)
            res.log('train/gen_gan_loss', gen_loss)
            res.log('train/gen_loss', l1_loss + gen_loss, prog_bar=True)
            return res

    def generator_step(self, batch):
        img, masked_img, mask, edges_mask = batch

        fine, coarse = self.gen(masked_img, mask, edges_mask)
        completed = self.get_completed_img(masked_img, mask, fine)

        # TODO: Add spatially discounted L1 loss
        l1_loss = F.l1_loss(coarse, img)
        l1_loss += F.l1_loss(fine, img)

        scores_fake = self.disc(completed, mask, edges_mask)
        gen_loss = -scores_fake.mean()

        return gen_loss, l1_loss

    def discriminator_step(self, batch):
        img, masked_img, mask, edges_mask = batch
        with torch.no_grad():
            fine, coarse = self.gen(masked_img, mask, edges_mask)
        completed = self.get_completed_img(masked_img, mask, fine)

        scores_fake = self.disc(completed, mask, edges_mask)
        hinge_fake = F.relu(1 + scores_fake).mean()
        scores_real = self.disc(img, mask, edges_mask)
        hinge_real = F.relu(1 - scores_real).mean()

        disc_loss = 0.5 * hinge_fake + 0.5 * hinge_real
        return disc_loss

    def validation_step(self, batch, batch_idx) -> pl.EvalResult:
        img, masked_img, mask, edges_mask = batch

        fine, coarse = self.gen(masked_img, mask, edges_mask)
        completed = self.get_completed_img(masked_img, mask, fine)
        scores_fake = self.disc(completed, mask, edges_mask)
        gen_loss = -scores_fake.mean()

        if batch_idx == 0:
            writer = self.logger.experiment
            writer.add_image('coarse', make_grid(coarse))
            writer.add_image('fine', make_grid(fine))
            writer.add_image('completed', make_grid(completed))

        res = pl.EvalResult(early_stop_on=gen_loss, checkpoint_on=gen_loss)
        res.log('val/gen_loss', gen_loss, prog_bar=True)
        res.log('val/l1_completed', F.l1_loss(completed, img), prog_bar=True)
        return res

    def configure_optimizers(self):
        disc_optim = optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
        gen_optim = optim.Adam(self.gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
        return disc_optim, gen_optim

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    @staticmethod
    def get_completed_img(masked_img, mask, fine):
        return masked_img + (1 - mask) * fine
