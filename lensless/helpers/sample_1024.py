from tqdm import tqdm
from lensless.helpers.diffusercam import DiffuserCam
from lensless.models.diffusion_model_64_1024 import UNet
from lensless.helpers.utils import extract, Variance
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import odak

DIFFUSERCAM_DIR = "/cs/student/projects1/2020/sonanuga/dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTEPS = 10


def normalize_tensor(x):
    return (x - x.min()) / (x.max() - x.min())


@torch.no_grad()
def p_sample(model, x, variance: Variance, t, t_index):
    betas_t = extract(variance.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        variance.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(variance.sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(variance.posterior_variance, t, x.shape)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * x


@torch.no_grad()
def p_sample_loop(model, diffused: torch.Tensor, variance, timesteps=10):
    device = next(model.parameters()).device

    b = 1

    img = diffused
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, variance, torch.full(
            (b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu())
    return imgs[-1]


@torch.no_grad()
def get_sample(model, image, variance, timesteps):
    return p_sample_loop(model, image, variance, timesteps)


def sample(testing_loader, model, variance, timesteps=TIMESTEPS, device=DEVICE):
    with torch.inference_mode():
        for i, data in enumerate(tqdm(testing_loader, 0)):
            diffused, _, lensed = data
            sample = get_sample(model, diffused.to(
                device), variance, timesteps)
            sample = normalize_tensor(sample)

            image = torch.permute(torch.squeeze(
                sample), (1, 2, 0))
            label = torch.permute(torch.squeeze(lensed), (1, 2, 0))

            odak.learn.tools.save_image(
                f"results_x0/{i}output.jpeg", image, cmin=0, cmax=1)

            odak.learn.tools.save_image(
                f"results_x0/{i}groundtruth.jpeg", label, cmin=0, cmax=1)


if __name__ == "__main__":
    network = UNet(in_channels=3, out_channels=3)
    checkpoint = torch.load("diffusion_modelx0_70_64_1024.pth")
    network.load_state_dict(checkpoint["model_state_dict"])
    variance = Variance(TIMESTEPS)
    network.eval()
    network.to(DEVICE)
    testing_loader = DataLoader(DiffuserCam(
        DIFFUSERCAM_DIR, training=False, testing=True).test_dataset)
    sample(testing_loader, network, variance, timesteps=TIMESTEPS)
