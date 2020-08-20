from pathlib import Path

import torch
from tqdm import tqdm
from torch import optim

from iminpaint.data.dataloader import dataloaders
from iminpaint.model import generator, discriminator


class Model:
    def __init__(self, img_folder):
        self.img_folder = img_folder

    def train(self, num_epochs, batch_size=16, num_workers=4, lr=1e-4, beta1=0.5,
              device=torch.device('cuda')):
        train_loader, val_loader = dataloaders.create_train_val_loader(self.img_folder,
                                                                       batch_size=batch_size,
                                                                       num_workers=num_workers)

        gen = generator.Generator().to(device)
        gen_optim = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

        disc = discriminator.Discriminator().to(device)
        disc_optim = optim.Adam(disc.parameters())

        for epoch in range(num_epochs):

            with tqdm(len(train_loader), desc='Train epoch {}'.format(epoch)) as pbar:
                for batch in train_loader:
                    print(batch)

                    pbar.update()

            with tqdm(len(val_loader), desc='Val epoch {}'.format(epoch)) as pbar:
                for batch in val_loader:
                    print(batch)

                    pbar.update()

        return self

    def predict(self):
        return self

    def load(self, path: Path):
        return self
