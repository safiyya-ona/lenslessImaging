from tqdm import tqdm
from lensless.helpers.utils import extract, Variance, normalize_tensor, transform_sample
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import invert
import odak

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


def sample_diffusion_model(model, collection, image_results_path, use_x0, device=DEVICE):
    testing_loader = DataLoader(collection.test_dataset)
    variance = Variance(TIMESTEPS)
    with torch.inference_mode():
        for i, data in enumerate(tqdm(testing_loader, 0)):
            diffused, _, lensed = data
            sample = get_sample(model, diffused.to(
                device), variance, TIMESTEPS)

            if use_x0:  # using x0 results in image with inverted colours, so reverted
                sample = invert(sample)

            image = transform_sample(sample)
            label = transform_sample(lensed)

            odak.learn.tools.save_image(
                f"{image_results_path}{i}output.png", image, cmin=0, cmax=1)

            odak.learn.tools.save_image(
                f"{image_results_path}{i}groundtruth.png", label, cmin=0, cmax=1)
