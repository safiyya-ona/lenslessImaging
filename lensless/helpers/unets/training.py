from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lensless.models.unets.simple_unet import UNet
from lensless.helpers.diffusercam import DiffuserCam

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 5
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(data_loader, model, optimizer, loss_fn):
    loop = tqdm(data_loader)

    for batch_idx, (_, propagated_data, targets) in enumerate(loop):
        optimizer.zero_grad()
        propagated = propagated_data.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(propagated)
        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
