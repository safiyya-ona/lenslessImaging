import torch
from torch.utils.data import DataLoader
import odak
from lensless.models.unets.simple_unet import UNet as SimpleUNet
from lensless.helpers.utils import transform_sample
from lensless.helpers.diffusercam import DiffuserCam
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(testing_loader, model, device=DEVICE):
    for i, data in enumerate(tqdm(testing_loader, 0)):
        diffused, propagated, lensed = data
        diffused = diffused.to(device)
        propagated = propagated.to(device)
        lensed = lensed.to(device)
        output = model(diffused)

        image = torch.permute(torch.squeeze(output), (2, 1, 0))
        label = torch.permute(torch.squeeze(lensed), (2, 1, 0))

        odak.learn.tools.save_image(
            f"results_results/{i}output.png", image, cmin=0, cmax=1)

        odak.learn.tools.save_image(
            f"results_results/{i}groundtruth.png", label, cmin=0, cmax=1)


def sample_unet(model, collection, image_results_path, device=DEVICE):
    testing_loader = DataLoader(collection.test_dataset)
    for i, data in enumerate(tqdm(testing_loader, 0)):
        diffused, propagated, lensed = data
        diffused = diffused.to(device)
        propagated = propagated.to(device)
        lensed = lensed.to(device)
        output = model(diffused)

        image = transform_sample(output)
        label = transform_sample(lensed)

        odak.learn.tools.save_image(
            f"{image_results_path}{i}output.png", image, cmin=0, cmax=1)

        odak.learn.tools.save_image(
            f"{image_results_path}{i}groundtruth.png", label, cmin=0, cmax=1)