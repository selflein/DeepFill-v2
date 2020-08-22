import os
from dataclasses import dataclass
from typing import Optional, Any, List

import hydra
import omegaconf
from omegaconf import SI
import pytorch_lightning as pl
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore

from iminpaint.model import DeepFill


@dataclass
class Model:
    use_contextual_attention: bool = False
    generator_width: float = 0.75
    disc_c_base: int = 64


@dataclass
class Data:
    path: str = f"{hydra.utils.get_original_cwd()}/data/datasets/flickr_dataset/training_imgs"
    edges_path: str = f"{hydra.utils.get_original_cwd()}/data/datasets/flickr_dataset/training_imgs_edges"
    batch_size: int = 4
    num_workers: int = 4


@dataclass
class Training:
    model: Model = Model()
    data: Data = Data()

# callbacks:
#   early_stopping:
#     class_name: pl.callbacks.EarlyStopping
#     params:
#       monitor: ${training.metric}
#       patience: 50
#       mode: min
#
#   model_checkpoint:
#     class_name: pl.callbacks.ModelCheckpoint
#     params:
#       monitor: ${training.metric}
#       save_top_k: 1
#       filepath: saved_models/


@dataclass
class EarlyStopping:
    _target_: str = "pl.callbacks.EarlyStopping"
    patience: int = 50
    mode: str = "min"


@dataclass
class ModelCheckpoint:
    _target_: str = "pl.callbacks.ModelCheckpoint"
    save_top_k: int = 1
    mode: str = "min"


@dataclass
class Trainer:
    min_epochs: int = 5


@dataclass
class Config:
    resume_checkpoint: Optional[str] = None
    trainer: Trainer = Trainer()
    training: Training = Training()
    early_stopping: EarlyStopping = EarlyStopping()
    model_checkpoint: ModelCheckpoint = ModelCheckpoint()


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)


@hydra.main(config_name='config')
def train(cfg: Config) -> None:
    print(cfg)
    model = DeepFill(hparams=cfg.training)

    # early_stopping = instantiate(cfg.early_stopping)
    # model_checkpoint = instantiate(cfg.model_checkpoint)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(logger=[tb_logger],
                         # early_stop_callback=early_stopping,
                         # checkpoint_callback=model_checkpoint,
                         # nb_sanity_val_steps=0,
                         **cfg.trainer)
    trainer.fit(model)


if __name__ == "__main__":
    train()
