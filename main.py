import argparse
import os
from pathlib import Path

from piq import psnr, multi_scale_ssim, LPIPS

import torch
import odak
from tqdm import tqdm
from lensless.helpers.diffusercam import DiffuserCam, region_of_interest
from lensless.helpers.unets.dtraining import train_unet
from lensless.helpers.diffusion_training import train_diffusion_model

from lensless.helpers.unets.evaluating import sample_unet
from lensless.helpers.sample import sample_diffusion_model

from lensless.models.unets.simple_unet import UNet
from lensless.models.diffusion_model import UNet as DiffusionModelUNet

SAVED_MODELS_PATH = "saved_models/"
IMAGE_RESULTS_PATH = "image_results/"

MODELS = {
    "unet": {"model": UNet, 'model_path': SAVED_MODELS_PATH + 'unet.pth', 'sample_path': IMAGE_RESULTS_PATH + 'unet/'},
    "diffusion_model_x0": {"model": DiffusionModelUNet,  "x0": True, 'model_path': SAVED_MODELS_PATH + 'diffusion_model_x0.pth', 'sample_path': IMAGE_RESULTS_PATH + 'diffusion_model_x0/'},
    "diffusion_model_eps": {"model": DiffusionModelUNet, "x0": False, 'model_path': SAVED_MODELS_PATH + 'diffusion_model_eps.pth', 'sample_path': IMAGE_RESULTS_PATH + 'diffusion_model_eps/'},
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
                        help="Enter flag to sample images from models")
    parser.add_argument("--results", action="store_true",
                        help="Enter flag to analyse model image reconstruction perfomance")
    parser.add_argument("--saved_models", type=Path, default=SAVED_MODELS_PATH,
                        help="Path to saved models")
    parser.add_argument("--image_results", type=Path,
                        default=IMAGE_RESULTS_PATH, help="Path to sampled images")

    return parser.parse_args()


def training_models(args):
    create_results_folder(SAVED_MODELS_PATH)
    collection = DiffuserCam(args.dataset, training=True, testing=False)
    for name in args.models:
        model = MODELS[name]
        if name == "unet":
            network = model["model"](3, 3).to(device)
            train_unet(network, collection, model["model_path"])
        elif name == "diffusion_model_x0":
            network = model["model"](3, 3).to(device)
            train_diffusion_model(
                network, collection, model["model_path"], use_x0=model["x0"])
        elif name == "diffusion_model_eps":
            network = model["model"](3, 3).to(device)
            train_diffusion_model(
                network, collection, model["model_path"], use_x0=model["x0"])


def sampling_models(args):
    create_results_folder(IMAGE_RESULTS_PATH)
    collection = DiffuserCam(args.dataset, training=False, testing=True)
    for name in args.models:
        model = MODELS[name]
        if name == "unet":
            network = model["model"](3, 3).to(device)
            network.load_state_dict(torch.load(model["model_path"]))
            network.eval()
            sample_unet(network, collection, create_results_folder(
                model["sample_path"]), device)
        elif name == "diffusion_model_x0":
            network = model["model"](3, 3).to(device)
            network.load_state_dict(torch.load(model["model_path"]))
            network.eval()
            sample_diffusion_model(network, collection,
                                   create_results_folder(model["sample_path"]), use_x0=model["x0"], device=device)
        elif name == "diffusion_model_eps":
            network = model["model"](3, 3).to(device)
            network.load_state_dict(torch.load(model["model_path"]))
            network.eval()
            sample_diffusion_model(network, collection,
                                   create_results_folder(model["sample_path"]), use_x0=model["x0"], device=device)


def model_results(args):
    for name in args.models:
        get_results(MODELS[name])


def get_results(model):
    """
    Returns a dictionary of results from the model in the results path
    """
    results = {'lpips': [], 'mse': [], 'psnr': [], 'ms-ssim': []}
    testing_images = sorted(os.listdir(model['sample_path']))
    mse_loss = torch.nn.MSELoss()
    lpips_loss = LPIPS()

    for i, _ in enumerate(tqdm(testing_images)):
        if i % 2 == 1:
            continue
        x = odak.learn.tools.load_image(
            model['sample_path'] + testing_images[i], normalizeby=255., torch_style=True)
        y = odak.learn.tools.load_image(
            model['sample_path'] + testing_images[i+1], normalizeby=255., torch_style=True)
        assert x is not None and y is not None
        y = region_of_interest(y.unsqueeze(0)).to(device)
        y_hat = region_of_interest(x.unsqueeze(0)).to(device)
        results['mse'].append(mse_loss(y_hat, y).item())
        results['psnr'].append(psnr(y_hat, y).item())
        results['ms-ssim'].append(multi_scale_ssim(y_hat, y).item())
        results['lpips'].append(lpips_loss(y_hat, y).item())
    average_results = {key: sum(value)/len(value)
                       for key, value in results.items()}
    print(average_results)
    return average_results


def create_results_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_main():
    print("Lensless Image Reconstruction using U-Net and Diffusion Models")

    args = parse_arguments()

    if args.train:
        training_models(args)
    if args.sample:
        sampling_models(args)
    if args.results:
        model_results(args)


if __name__ == "__main__":
    run_main()
