import argparse
import os
import time
import timeit
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from piq import psnr, multi_scale_ssim, LPIPS

import torch
import torchvision
from torch.utils.data import DataLoader
import tqdm
from helpers.diffusercam import DiffuserCam

MODELS = {
    "full-image": {"width": 5, "depth": 5}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a model for lensless image reconstruction")
    parser.add_argument("--models", nargs='*', default=MODELS,
                        choices=MODELS.keys(), help="Model to train neural network")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train neural network")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--train", action="store_true",
                        help="Path to train dataset")

    return parser.parse_args()


def training_models(args):
    collection = DiffuserCam(args.dataset)
    for name in args.models:
        model = MODELS[name]
        print(
            f"Training model {name} with width {model['width']} and depth {model['depth']}")


def run_main():
    print("Lensless Imaging using Small Convolutional Kernels")

    args = parse_arguments()

    if args.train:
        print(
            f"Training the neural network with model {args.models} for {args.epochs} epochs")
        print(f"Dataset Path: {args.dataset}")
        training_models(args)


if __name__ == "__main__":
    run_main()
