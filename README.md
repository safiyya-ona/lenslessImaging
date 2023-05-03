# Diffusion Models for Diffuser Cameras

For my final year project for UCL Computer Science, I investigated the use of Diffusion Models for image reconstruction of lensless cameras, using the DiffuserCam data set.

The dataset can be found through this link: [DiffuserCam Dataset](https://waller-lab.github.io/LenslessLearning/dataset.html)

For all commands, the path to the dataset will be "path_to_dataset"

## Commands

Saved models can be found through this link: [Pretrained Models](https://liveuclac-my.sharepoint.com/:f:/g/personal/zcabson_ucl_ac_uk/EoZFg4unWXRJhE6FOkNkcHkBZGEFFDFwWWAh7rftQquhDw?e=HmhiiN)

By default, all models are selected if the flag is ommitted. You can select one model by adding the flag --models and its name, e.g.

    --models "diffusion_model_x0"

### Sampling Images

    python main.py --dataset "path_to_dataset" --models --sample

### Training Images

    python main.py --dataset "path_to_dataset" --models --train

### Getting average image results

    python main.py --dataset "path_to_dataset" --models --results
