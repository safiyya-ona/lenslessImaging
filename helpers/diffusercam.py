from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from pathlib import Path
from PIL import Image


def transform(sample):
    """Transforms a sample to a tensor"""

    return to_tensor(sample)


class DiffuserCamDataset(Dataset):
    """Dataset for the DiffuserCam dataset, transforming images to tensors"""

    def __init__(self, diffuers_images, ground_truth_images, transform=transform):
        self.diffuser_images = diffuers_images
        self.ground_truth_images = ground_truth_images
        self.transform = transform

    def __len__(self):
        return len(self.diffuser_images)

    def __getitem__(self, idx):
        diffuser_image = self.diffuser_images[idx]
        ground_truth_image = self.ground_truth_images[idx]

        x, y = None, None
        if self.transform:
            x = self.transform(diffuser_image)
            y = self.transform(ground_truth_image)

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

        diffuser_images, ground_truth_images = [], []
        for label in labels:
            diffuser_image = path / "diffuser_images" / \
                label.replace(".jpg.tiff", ".npy")
            ground_truth_image = path / "ground_truth_lensed" / \
                label.replace(".jpg.tiff", ".npy")
            if diffuser_image.exists() and ground_truth_image.exists():
                diffuser_images.append(diffuser_image)
                ground_truth_images.append(ground_truth_image)
            else:
                print(
                    f"Image {label} not found in dataset. No file named {diffuser_image} or {ground_truth_image}")
        return diffuser_images, ground_truth_images
