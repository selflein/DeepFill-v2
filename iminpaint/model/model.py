from pathlib import Path

import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F

from iminpaint.model.generator import Generator
from iminpaint.data.dataloader import dataloaders
from iminpaint.model.discriminator import Discriminator


class Model:
    def __init__(self):
        pass

    def train(self, img_folder, edges_folder, num_epochs, batch_size=16, num_workers=4, lr=1e-4, beta1=0.5,
              device=torch.device('cuda')):
        train_loader, val_loader = dataloaders.create_train_val_loader(
            img_folder,
            edges_folder,
            batch_size=batch_size,
            num_workers=num_workers
        )

        gen = Generator(width=1, use_contextual_attention=False).to(device)
        gen_optim = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

        disc = Discriminator(c_base=64).to(device)
        disc_optim = optim.Adam(disc.parameters())

        for epoch in range(num_epochs):

            with tqdm(len(train_loader), desc='Train epoch {}'.format(epoch)) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    img, masked_img, mask, edges_mask = batch

                    # Update discriminator
                    disc_optim.zero_grad()
                    with torch.no_grad():
                        coarse, fine = gen(masked_img, mask, edges_mask)

                    scores_fake = disc(fine, mask, edges_mask)
                    scores_real = disc(img, mask, edges_mask)

                    disc_loss.backward()
                    disc_optim.step()

                    # Update generator
                    if batch_idx % 5 == 0:
                        gen_optim.zero_grad()

                        coarse, fine = gen(masked_img, mask, edges_mask)
                        scores_fake = disc(fine, mask, edges_mask)

                        # TODO: Add spatially discounted L1 loss
                        gen_loss = F.l1_loss(coarse, img)

                        gen_loss.backward()
                        gen_optim.step()

                    pbar.update()

            with tqdm(len(val_loader), desc='Val epoch {}'.format(epoch)) as pbar:
                for batch in val_loader:
                    img, masked_img, mask, edges_mask = batch

                    pbar.update()

        return self

    def predict(self):
        return self

    def load(self, path: Path):
        return self
