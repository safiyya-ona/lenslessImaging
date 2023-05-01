import argparse
import os
import time
import timeit
from pathlib import Path

from piq import psnr, multi_scale_ssim, LPIPS

import torch
import torchvision
from torch.utils.data import DataLoader
import tqdm
from lensless.helpers.diffusercam import DiffuserCam, region_of_interest
from lensless.helpers.unets.dtraining import train_unet
from lensless.helpers.diffusion_training import train_diffusion_model
from lensless.models.unets.simple_unet import UNet
from lensless.models.diffusion_model import UNet as DiffusionModelUNet

SAVED_MODELS_PATH = "saved_models/"
IMAGE_RESULTS_PATH = "image_results/"

MODELS = {
    "unet": {"model": UNet, 'results_path': 'saved_models/unet.pth'},
    "diffusion_model_x0": {"model": DiffusionModelUNet,  "x0": True, 'results_path': SAVED_MODELS_PATH + '/diffusion_model_x0.pth'},
    "diffusion_model_eps": {"model": DiffusionModelUNet, "x0": False, 'results_path': SAVED_MODELS_PATH + 'saved_models/diffusion_model_eps.pth'},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a model for lensless image reconstruction")
    parser.add_argument("--models", nargs='*', default=MODELS,
                        choices=MODELS.keys(), help="Select a model to train")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--train", action="store_true",
                        help="Enter flag to train a model")
    parser.add_argument("--sample", action="store_true",
                        help="Enter flag to sample from a model")
    parser.add_argument("--saved_models", type=Path, default=SAVED_MODELS_PATH,
                        help="Path to saved models")
    parser.add_argument("--image_results", type=Path,
                        default=IMAGE_RESULTS_PATH, help="Path to image results")

    return parser.parse_args()


def training_models(args):
    create_results_folder(SAVED_MODELS_PATH)
    collection = DiffuserCam(args.dataset, training=True, testing=False)
    for name in args.models:
        model = MODELS[name]
        if name == "unet":
            network = model["model"](3, 3).to(device)
            train_unet(network, collection, model["results_path"])
        elif name == "diffusion_model_x0":
            network = model["model"](3, 3).to(device)
            train_diffusion_model(
                network, collection, model["results_path"], use_x0=model["x0"])
        elif name == "diffusion_model_eps":
            network = model["model"](3, 3).to(device)
            train_diffusion_model(
                network, collection, model["results_path"], use_x0=model["x0"])


# def model_results(model, dataset_path, results_path):
#     """
#     Returns a dictionary of results from the model in the results path
#     """
#     results = {'lpips': [], 'mse': [], 'psnr': [], 'ms-ssim': []}
#     testing_images = DiffuserCam(dataset_path, training=False, testing=True)
#     mse_loss = torch.nn.MSELoss()
#     lpips_loss = LPIPS(weights=VGG16_Weights.IMAGENET1K_V1)

#     for i in tqdm.trange(0, len(testing_images)):
#         x, y = testing_images[i]
#         x = x.unsqueeze(0)
#         y = region_of_interest(y.unsqueeze(0)).to(device)
#         y_hat = region_of_interest(testing_images.unsqueeze(0)).to(device)
#         results['mse'].append(mse_loss(y_hat, y).item())
#         results['psnr'].append(psnr(y_hat, y).item())
#         results['ms-ssim'].append(multi_scale_ssim(y_hat, y).item())
#         results['lpips'].append(lpips_loss(y_hat, y).item())
#     average_results = {key: sum(value)/len(value)
#                        for key, value in results.items()}
#     return average_results

def create_results_folder(path):
    os.makedirs(path, exist_ok=True)


def run_main():
    print("Lensless Image Reconstruction using U-Net and Diffusion Models")

    args = parse_arguments()

    if args.train:
        training_models(args)
    # if args.sample:
    #     sampling_models(args)
    # if args.results:
    #     model_results(args)


if __name__ == "__main__":
    run_main()
