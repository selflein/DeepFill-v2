import os
from typing import Optional, Any, List
from dataclasses import dataclass, field

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from hydra.core.config_store import ConfigStore

from iminpaint.model import DeepFill


@dataclass
class Model:
    use_contextual_attention: bool = True
    generator_width: float = 0.75
    disc_c_base: int = 64


@dataclass
class Data:
    path: str = f"data/datasets/flickr_dataset/training_imgs"
    edges_path: str = f"data/datasets/flickr_dataset/training_imgs_edges"
    batch_size: int = 4
    num_workers: int = 4
    train_percentage: float = 0.95


@dataclass
class Training:
    model: Model = Model()
    data: Data = Data()


@dataclass
class EarlyStopping:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    patience: int = 50
    mode: str = "min"


@dataclass
class ModelCheckpoint:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    save_top_k: int = 1
    mode: str = "min"
    save_last: bool = True
    filepath: str = 'checkpoints/best.ckpt'


@dataclass
class Trainer:
    min_epochs: int = 5
    gpus: List[int] = field(default_factory=lambda: [0])


@dataclass
class TestingTrainer(Trainer):
    # limit_train_batches: float = 0.01
    # limit_val_batches: float = 0.01
    # track_grad_norm: float = 2.0
    overfit_batches: int = 1
    min_epochs: int = 1000
    check_val_every_n_epoch: int = 20


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: [{'trainer': 'trainer'}])
    resume_from_checkpoint: Optional[str] = None
    trainer: Trainer = Trainer()
    training: Training = Training()
    early_stopping: EarlyStopping = EarlyStopping()
    model_checkpoint: ModelCheckpoint = ModelCheckpoint()


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)
cs.store(group='trainer', name='trainer', node=Trainer)
cs.store(group='trainer', name='testing_trainer', node=TestingTrainer)


@hydra.main(config_name='config')
def train(cfg: Config) -> None:
    print(omegaconf.OmegaConf.to_yaml(cfg))
    model = DeepFill(hparams=cfg.training)

    early_stopping = instantiate(cfg.early_stopping)
    model_checkpoint = instantiate(cfg.model_checkpoint)

    resume_model = None
    if cfg.resume_from_checkpoint is not None:
        resume_model = to_absolute_path(cfg.resume_from_checkpoint)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(logger=tb_logger,
                         resume_from_checkpoint=resume_model,
                         early_stop_callback=early_stopping,
                         checkpoint_callback=model_checkpoint,
                         **cfg.trainer)
    trainer.fit(model)


if __name__ == "__main__":
    train()
