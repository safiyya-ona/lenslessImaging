from tqdm import tqdm
from lensless.helpers.diffusercam import DiffuserCam
from lensless.models.diffusion_model_old import UNet
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import odak

DIFFUSERCAM_DIR = "/cs/student/projects1/2020/sonanuga/dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def normalize_tensor(x):
    return (x - x.min()) / (x.max() - x.min())


def initialise_sample_variables(timesteps=10):
    # define beta schedule
    betas = cosine_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * \
        (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return (betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)


@torch.no_grad()
def p_sample(model, x, s_v, t, t_index):
    betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance = s_v
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 but save all images:


@torch.no_grad()
def p_sample_loop(model, diffused: torch.Tensor, timesteps=10):
    device = next(model.parameters()).device

    b = 1

    img = diffused
    imgs = []

    for i in reversed(range(0, timesteps)):
        s_v = initialise_sample_variables(timesteps)
        img = p_sample(model, img, s_v, torch.full(
            (b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu())
    return imgs[-1]


@torch.no_grad()
def get_sample(model, image, timesteps):
    return p_sample_loop(model, image, timesteps)


def sample(testing_loader, model, timesteps=10, device=DEVICE):
    with torch.inference_mode():
        for i, data in enumerate(tqdm(testing_loader, 0)):
            diffused, _, lensed = data
            sample = get_sample(model, diffused.to(device), timesteps)
            sample = normalize_tensor(sample)
            # t = torch.full((1,), timesteps, device=device, dtype=torch.long)
            # sample = model(diffused.to(device), t.to(device))

            image = torch.permute(torch.squeeze(
                sample), (1, 2, 0))
            label = torch.permute(torch.squeeze(lensed), (1, 2, 0))

            odak.learn.tools.save_image(
                f"results_10/{i}output.jpeg", image, cmin=0, cmax=1)

            odak.learn.tools.save_image(
                f"results_10/{i}groundtruth.jpeg", label, cmin=0, cmax=1)


if __name__ == "__main__":
    network = UNet(in_channels=3, out_channels=3)  # for rgb channels
    # checkpoint = torch.load("diffusion_model_100_10_24-512t.pth")
    # network.load_state_dict(checkpoint["model_state_dict"])
    checkpoint = torch.load("diffusion_model_10.pth")
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    network.to(DEVICE)
    testing_loader = DataLoader(DiffuserCam(
        DIFFUSERCAM_DIR, training=False, testing=True).test_dataset)
    sample(testing_loader, network, timesteps=10)
