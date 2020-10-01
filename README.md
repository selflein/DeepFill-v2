## DeepFill v2 PyTorch
**WIP** Unofficial implementation of "[Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)".

### Setup
Install conda environment
```
conda env create -f environment.yml
conda activate iminpaint
```


### Training
Hydra together with PyTorch-Lightning is used for implementation. To start default training:
```
python train.py
```
and a folder will be created with Tensorboard logs and checkpoints in `./outputs`.

### TODOs
* [x] Implement Contextual Attention & respective network branch 
* [ ] Add support for Places2 and CelebA-HQ faces datasets used in the paper
* [ ] Add pre-trained models 
 