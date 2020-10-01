## DeepFill v2 PyTorch
**WIP** Unofficial implementation of "[Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)".

### Setup
* Install conda environment
    ```
    conda env create -f environment.yml
    conda activate iminpaint
    ```
* Setup data
    * Put the training dataset into one folder of images, e.g., `data/training_imgs`
    * Run the edge extraction
        ```
        python iminpaint/data/scripts/generate_edge_masks --input_folder data/training_imgs --output_folder some_folder --prototxt path_to_prototxt --caffemodel path_to_model
        ```
      Download the prototxt from [here](https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt) and the caffe model from [here](http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel).   
      Optionally, one can add the `--visualize` flag to plot an example edge output.

    * Adjust the paths to the folder with training images and edges in the training script `train.py` in the `Data` dataclass.
    
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
 