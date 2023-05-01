from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lensless.models.unets.simple_unet import UNet
from lensless.models.unets.attention_unet import AttentionUNet
from lensless.helpers.diffusercam import DiffuserCam

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 5
NUM_EPOCHS = 10
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 270
IMAGE_WIDTH = 480
PIN_MEMORY = True
LOAD_MODEL = False
DIFFUSERCAM_DIR = "/cs/student/projects1/2020/sonanuga/dataset"


def run_epoch(data_loader, model, optimizer, loss_fn):
    loop = tqdm(data_loader)
    loss = None
    for batch_idx, (diffuser_data, propagated_data, targets) in enumerate(loop):
        optimizer.zero_grad()
        propagated = diffuser_data.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(propagated)
        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
    return loss


def train_unet(model, collection, save_model_path):
    data_loader = DataLoader(
        collection.train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        loss = run_epoch(data_loader, model, optimizer, loss_fn)
    torch.save(model.state_dict(), save_model_path)
