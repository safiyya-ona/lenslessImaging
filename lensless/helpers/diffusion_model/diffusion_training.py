from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lensless.models.diffusion_model.diffusion_model import UNet
from lensless.helpers.diffusercam import DiffuserCam
from lensless.helpers.utils import extract, Variance

# Hyperparameters
LEARNING_RATE = 0.0002
LEARNING_RATE_NO_SCHEDULER = 0.0001
MAX_LEARNING_RATE = 0.01
BATCH_SIZE = 5
NUM_EPOCHS = 100
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True
TIMESTEPS = 10


def add_noise(clean_image, t, variance: Variance, noise=None):
    """
    Add noise to the image based on the timestep t.
    """
    if noise is None:
        noise = torch.randn_like(clean_image)

    sqrt_alphas_cumprod = variance.get_alphas_cumprod()
    sqrt_one_minus_alphas_cumprod = variance.get_one_minus_alphas_cumprod()
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, clean_image.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, clean_image.shape
    )

    return sqrt_alphas_cumprod_t * clean_image + sqrt_one_minus_alphas_cumprod_t * noise


def run_epoch(
    data_loader, model, optimizer, scheduler, loss_fn, variance, use_x0=False
):
    loop = tqdm(data_loader)
    loss = None
    for _, (diffuser_data, _, targets) in enumerate(loop):
        optimizer.zero_grad()
        diffused_images = diffuser_data.to(DEVICE)
        clean_images = targets.to(DEVICE)
        t = torch.randint(0, TIMESTEPS, (diffuser_data.shape[0],), device=DEVICE).long()

        noisy_images = add_noise(clean_images, t, variance, diffused_images).to(DEVICE)

        noise_predictions = model(noisy_images, t)
        loss = (
            loss_fn(clean_images, noise_predictions)
            if use_x0
            else loss_fn(diffused_images, noise_predictions)
        )

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
        else:
            loop.set_postfix(loss=loss.item())
    return loss


def train_diffusion_model(
    model,
    collection,
    results_path,
    use_x0=False,
    timesteps=TIMESTEPS,
    epochs=NUM_EPOCHS,
):
    loss_fn = nn.MSELoss()
    training_loader: DataLoader = DataLoader(
        collection.train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LEARNING_RATE,
        steps_per_epoch=len(training_loader),
        epochs=NUM_EPOCHS,
    )

    scheduler = None

    variance = Variance(timesteps)

    for epoch in range(1, epochs + 1):
        loss = run_epoch(
            training_loader, model, optimizer, scheduler, loss_fn, variance, use_x0
        )
        torch.save(model.state_dict(), results_path)
        print(f"Epoch {epoch} loss: {loss.item()}")


def train_diffusion_model_no_scheduler(
    model,
    collection,
    results_path,
    use_x0=False,
    timesteps=TIMESTEPS,
    epochs=NUM_EPOCHS,
):
    loss_fn = nn.MSELoss()
    training_loader: DataLoader = DataLoader(
        collection.train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_NO_SCHEDULER)

    scheduler = None

    variance = Variance(timesteps)

    for epoch in range(1, epochs + 1):
        loss = run_epoch(
            training_loader, model, optimizer, scheduler, loss_fn, variance, use_x0
        )
        torch.save(model.state_dict(), results_path)
        print(f"Epoch {epoch} loss: {loss.item()}")
