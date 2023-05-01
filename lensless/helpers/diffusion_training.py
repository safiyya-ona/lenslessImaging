from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lensless.models.diffusion_model_1024 import UNet
from lensless.helpers.diffusercam import DiffuserCam
from lensless.helpers.utils import extract, Variance

from PIL import Image

# Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 5
NUM_EPOCHS = 100
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 270
IMAGE_WIDTH = 480
PIN_MEMORY = True
LOAD_MODEL = False
DIFFUSERCAM_DIR = "/cs/student/projects1/2020/sonanuga/dataset"
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


def run_epoch(data_loader, model, optimizer, loss_fn, variance):
    loop = tqdm(data_loader)
    running_loss = 0.0
    loss = None
    for batch_idx, (diffuser_data, _, targets) in enumerate(loop):
        optimizer.zero_grad()
        lensed_images = diffuser_data.to(DEVICE)
        clean_images = targets.to(DEVICE)
        t = torch.randint(
            0, TIMESTEPS, (diffuser_data.shape[0],), device=DEVICE).long()

        noisy_images = add_noise(
            clean_images, t, variance, lensed_images).to(DEVICE)

        noise_predictions = model(noisy_images, t)
        loss = loss_fn(lensed_images, noise_predictions)

        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    return loss


def train(timesteps=TIMESTEPS):
    model = UNet(3, 3).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    diffuser_collection: "DiffuserCam" = DiffuserCam(DIFFUSERCAM_DIR)
    training_loader: DataLoader = DataLoader(
        diffuser_collection.train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    loss_history = []
    variance = Variance(timesteps)

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = run_epoch(training_loader, model, optimizer, loss_fn, variance)
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history,
                'variance': variance
            }, f"diffusion_model_{epoch}_real1024.pth")


def main():
    train()


if __name__ == "__main__":
    main()
