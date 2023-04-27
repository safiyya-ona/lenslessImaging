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


def train(data_loader, model, optimizer, loss_fn):
    loop = tqdm(data_loader)

    for batch_idx, (diffuser_data, propagated_data, targets) in enumerate(loop):
        optimizer.zero_grad()
        propagated = propagated_data.to(DEVICE)
        targets = targets.to(DEVICE)

        predictions = model(propagated)
        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())


def main():
    model = AttentionUNet(3, 3).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    diffuser_collection: "DiffuserCam" = DiffuserCam(DIFFUSERCAM_DIR)
    training_loader: DataLoader = DataLoader(
        diffuser_collection.train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    for epoch in range(NUM_EPOCHS):
        train(training_loader, model, optimizer, loss_fn)

    torch.save(model.state_dict(), "conditional_unet_prop10.pth")


if __name__ == "__main__":
    main()
