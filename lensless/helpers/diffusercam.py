import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, resize, rgb_to_grayscale
from tqdm import tqdm
from lensless.helpers.beam_propagation import Simulation


SIZE = 270, 480
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transform(sample):
    """Transforms a sample to a tensor"""
    image = to_tensor(sample)
    image = resize(image, SIZE)
    return image


class DiffuserCamDataset(Dataset):
    """Dataset for the DiffuserCam dataset, transforming images to tensors"""

    def __init__(self, diffuers_images, ground_truth_images, transform=transform, use_diffuse_transform=False):
        self.diffuser_images = diffuers_images
        self.ground_truth_images = ground_truth_images
        self.transform = transform
        self.simulator = Simulation()

    def __len__(self):
        return len(self.diffuser_images)

    def __getitem__(self, idx):
        diffuser_image = self.diffuser_images[idx]
        ground_truth_image = self.ground_truth_images[idx]

        x, y = None, None
        if self.use_diffuse_transform:
            # when testing for grayscale images
            x = rgb_to_grayscale(
                self.diffuser_transform(np.load(ground_truth_image)))
            y = rgb_to_grayscale(self.transform(np.load(ground_truth_image)))
        else:
            # when testing for RGB images
            x = self.transform(np.load(diffuser_image))
            y = self.transform(np.load(ground_truth_image))

        return x, y


class DiffuserCam:
    def __init__(self, path) -> None:
        self.path = Path(path)

        self.psf = Image.open(self.path / "psf.tiff")

        training_diffused, training_ground_truth = self.get_dataset_images(
            self.path, "dataset_train.csv")
        testing_diffused, testing_ground_truth = self.get_dataset_images(
            self.path, "dataset_test.csv")

        self.train_dataset = DiffuserCamDataset(
            training_diffused, training_ground_truth)
        self.test_dataset = DiffuserCamDataset(
            testing_diffused, testing_ground_truth)

    def get_dataset_images(self, path, filename):
        """Get the images from the dataset path given on input"""
        with open(path / filename) as f:
            labels = f.read().split()

        diffuser_images, ground_truth_images, propagated_diffused_images = [], [], []
        simulator = None
        for label in tqdm(labels):
            diffuser_image = path / "diffuser_images" / \
                label.replace(".jpg.tiff", ".npy")
            ground_truth_image = path / "ground_truth_lensed" / \
                label.replace(".jpg.tiff", ".npy")
            propagated_diffused_image = path / "propagated_diffused_images" / \
                label.replace(".jpg.tiff", ".npy")
            if diffuser_image.exists() and ground_truth_image.exists():
                diffuser_images.append(diffuser_image)
                ground_truth_images.append(ground_truth_image)
                if not propagated_diffused_image.exists():
                    new_image = to_tensor(
                        np.load(ground_truth_image)).to(DEVICE)
                    if simulator is None:
                        simulator = Simulation(resolution=new_image.shape[1:])
                    new_propagated_image = simulator.diffuse_image(new_image)
                    np.save(propagated_diffused_image,
                            new_propagated_image.cpu())
                propagated_diffused_images.append(propagated_diffused_image)
            else:
                print(
                    f"Image {label} not found in dataset. No file named {diffuser_image} or {ground_truth_image}")
        return diffuser_images, ground_truth_images
