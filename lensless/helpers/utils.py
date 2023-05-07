import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate

# code inspired by https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb


def _cosine_beta_schedule(timesteps, s=0.008):
    """
    Implements cosine schedule from https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    """
    Extracts the appropriate t index for a batch of indices
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def normalize_tensor(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())


def transform_sample(sample: torch.Tensor):
    """
    The DiffuserCam dataset stores images as BGR and upside down. This function returns the image in RGB and right side up.
    """
    transformed_sample = torch.permute(torch.squeeze(rotate(sample, 180)), (1, 2, 0))

    return torch.flip(transformed_sample, [1, 2])


class Variance:
    """
    Stores necessary constants needed for forward process and sampling images from a diffusion model
    """

    def __init__(self, timesteps) -> None:
        self.betas = _cosine_beta_schedule(timesteps=timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def get_alphas_cumprod(self):
        return self.alphas_cumprod

    def get_one_minus_alphas_cumprod(self):
        return self.sqrt_one_minus_alphas_cumprod

    def get_betas(self):
        return self.betas

    def get_sqrt_recip_alphas(self):
        return self.sqrt_recip_alphas

    def get_posterior_variance(self):
        return self.posterior_variance

    def get_sqrt_one_minus_alphas_cumprod(self):
        return self.sqrt_one_minus_alphas_cumprod
