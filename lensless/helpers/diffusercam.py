import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, resize
from tqdm import tqdm
from lensless.helpers.beam_propagation import Simulation


SIZE = 270, 480
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def region_of_interest(x):
    return x[..., 60:270, 60:440]


def transform(sample):
    """Transforms a sample to a tensor"""
    image = to_tensor(sample)
    image = resize(image, SIZE)
    return image


def transform_propagated(sample):
    """Transforms a sample to a tensor"""
    image = to_tensor(sample)
    image = resize(image.permute(1, 2, 0), SIZE)
    return image


class DiffuserCamDataset(Dataset):
    """Dataset for the DiffuserCam dataset, transforming images to tensors"""

    def __init__(self, diffuser_images, propagated_images, ground_truth_images, transform=transform):
        self.diffuser_images = diffuser_images
        self.ground_truth_images = ground_truth_images
        self.propagated_images = propagated_images
        self.transform = transform

    def __len__(self):
        return len(self.diffuser_images)

    def __getitem__(self, idx):
        diffuser_image = self.diffuser_images[idx]
        ground_truth_image = self.ground_truth_images[idx]
        propagated_image = self.propagated_images[idx]

        x, y, z = None, None, None
        if self.transform:
            x = self.transform(np.load(diffuser_image))
            y = transform_propagated(
                np.load(propagated_image))
            z = self.transform(np.load(ground_truth_image))

        return x, y, z


class DiffuserCam:
    def __init__(self, path, training=True, testing=False) -> None:
        self.path = Path(path)

        if training:
            training_diffused, training_propagated, training_ground_truth = self.get_dataset_images(
                self.path, "dataset_train.csv")
            self.train_dataset = DiffuserCamDataset(
                training_diffused, training_propagated, training_ground_truth)

        if testing:
            testing_diffused, testing_propagated, testing_ground_truth = self.get_dataset_images(
                self.path, "dataset_test.csv")
            self.test_dataset = DiffuserCamDataset(
                testing_diffused, testing_propagated, testing_ground_truth)

    def get_dataset_images(self, path, filename):
        """Get the images from the dataset path given on input"""
        with open(path / filename) as f:
            labels = f.read().split()
        diffuser_images, ground_truth_images, propagated_diffused_images = [], [], []
        simulator = None
        for label in tqdm(labels, desc="Loading images"):
            diffuser_image = path / "diffuser_images" / \
                label.replace(".jpg.tiff", ".npy")
            ground_truth_image = path / "ground_truth_lensed" / \
                label.replace(".jpg.tiff", ".npy")
            propagated_diffused_image = path / "propagated_rgb_diffused_images" / \
                label.replace(".jpg.tiff", ".npy")
            if diffuser_image.exists() and ground_truth_image.exists():
                if not propagated_diffused_image.exists():
                    new_image = to_tensor(
                        np.load(ground_truth_image)).to(DEVICE)
                    if simulator is None:
                        simulator = Simulation(resolution=new_image.shape[1:])
                    new_propagated_image = simulator.diffuse_rgb_image(
                        new_image)
                    np.save(propagated_diffused_image,
                            new_propagated_image.cpu(), allow_pickle=False)
                else:
                    try:
                        prop_image = np.load(propagated_diffused_image)
                        if prop_image.shape != (3, 270, 480):
                            new_image = to_tensor(
                                np.load(ground_truth_image)).to(DEVICE)
                            if simulator is None:
                                simulator = Simulation(
                                    resolution=new_image.shape[1:])
                                new_propagated_image = simulator.diffuse_rgb_image(
                                    new_image)
                                if new_propagated_image.shape != (3, 270, 480):
                                    print(new_propagated_image.shape,
                                          new_image.shape, propagated_diffused_image)
                                    continue
                                else:
                                    np.save(propagated_diffused_image,
                                            new_propagated_image.cpu(), allow_pickle=False)

                                continue
                    except ValueError:
                        print(f"Image {label} is using pickle.")
                        new_image = to_tensor(
                            np.load(ground_truth_image)).to(DEVICE)
                        if simulator is None:
                            simulator = Simulation(
                                resolution=new_image.shape[1:])
                        new_propagated_image = simulator.diffuse_rgb_image(
                            new_image)
                        np.save(propagated_diffused_image,
                                new_propagated_image.cpu(), allow_pickle=False)
                    diffuser_images.append(diffuser_image)
                    ground_truth_images.append(ground_truth_image)
                    propagated_diffused_images.append(
                        propagated_diffused_image)
            else:
                print(
                    f"Image {label} not found in dataset. No file named {diffuser_image} or {ground_truth_image}")
        return diffuser_images, propagated_diffused_images, ground_truth_images
