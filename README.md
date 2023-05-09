# Learned Diffusion Processes for Lensless Cameras

For my final year project for UCL Computer Science, I investigated the use of processes from diffusion models for image reconstruction of lensless cameras, using the DiffuserCam data set.

The dataset can be found through this link: [DiffuserCam Dataset](https://waller-lab.github.io/LenslessLearning/dataset.html)

For all commands, the path to the dataset will be "path_to_dataset"

## Setup

Using the environment.yml file, you can run

    conda env create -f environment.yml

Or manually install using the commands below

    conda create -n lensless-diffusion pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
    conda activate lensless-diffusion
    pip install tqdm einops piq odak

## Commands for training and evaluation

Saved models can be found through this link: [Pretrained Models](https://liveuclac-my.sharepoint.com/:f:/g/personal/zcabson_ucl_ac_uk/Ej6XsdayLeRBsxO2p0V5eOUBsX9xyfb5c_mxJx5JvMoLPQ?e=flLKQ3)

By default, all models are selected if the flag is ommitted. You can select one model by adding the flag --models and its name, e.g.

    --models "residual_diffusion_model_x0"

### Sampling Images

    python main.py --dataset "path_to_dataset" --sample --models "residual_diffusion_model_x0"

### Training Images

    python main.py --dataset "path_to_dataset" --train --models "residual_diffusion_model_x0"

### Getting average image results

    python main.py --dataset "path_to_dataset" --results --models "residual_diffusion_model_x0"
