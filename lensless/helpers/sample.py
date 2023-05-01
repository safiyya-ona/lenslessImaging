from tqdm import tqdm
from lensless.helpers.diffusercam import DiffuserCam
from lensless.models.diffusion_model import UNet
from lensless.helpers.utils import extract, Variance, normalize_tensor, transform_sample
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rotate
import odak

DIFFUSERCAM_DIR = "/cs/student/projects1/2020/sonanuga/dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTEPS = 10


@torch.no_grad()
def p_sample(model, x, variance: Variance, t, t_index):
    betas_t = extract(variance.get_betas(), t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        variance.get_sqrt_one_minus_alphas_cumprod(), t, x.shape
    )
    sqrt_recip_alphas_t = extract(variance.get_sqrt_recip_alphas(), t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(
            variance.get_posterior_variance(), t, x.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * x


@torch.no_grad()
def p_sample_loop(model, diffused: torch.Tensor, variance, timesteps=TIMESTEPS):
    device = next(model.parameters()).device

    batch_size = 1

    img = diffused
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, variance, torch.full(
            (batch_size,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu())
    return imgs[-1]


@torch.no_grad()
def get_sample(model, image, variance, timesteps):
    sample = p_sample_loop(model, image, variance, timesteps)
    return normalize_tensor(sample)


def sample(testing_loader, model, variance, timesteps=TIMESTEPS, device=DEVICE):
    with torch.inference_mode():
        for i, data in enumerate(tqdm(testing_loader, 0)):
            diffused, _, lensed = data
            sample = get_sample(model, diffused.to(
                device), variance, timesteps)

            image = transform_sample(sample)
            label = transform_sample(lensed)

            odak.learn.tools.save_image(
                f"results_x0/{i}output.jpeg", image, cmin=0, cmax=1)

            odak.learn.tools.save_image(
                f"results_x0/{i}groundtruth.jpeg", label, cmin=0, cmax=1)


def sample_image(model, diffused_image, timesteps=TIMESTEPS):
    diffused_image = diffused_image.to(model.device)
    variance = Variance(timesteps)
    sample = get_sample(model, diffused_image, variance, timesteps)
    return sample


if __name__ == "__main__":
    network = UNet(in_channels=3, out_channels=3)
    checkpoint = torch.load("diffusion_modelx0_90_64_1024.pth")
    network.load_state_dict(checkpoint["model_state_dict"])
    variance = Variance(TIMESTEPS)
    network.eval()
    network.to(DEVICE)
    testing_loader = DataLoader(DiffuserCam(
        DIFFUSERCAM_DIR, training=False, testing=True).test_dataset)
    sample(testing_loader, network, variance, timesteps=TIMESTEPS)
