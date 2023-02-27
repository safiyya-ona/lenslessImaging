import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt
import numpy as np
import odak
from tqdm import tqdm

from helpers.diffusercam import DiffuserCam
from helpers.beam_propagation import Simulation

BATCH_SIZE = 20
IMAGE_SIZE = 270, 480

diffuser_collection: "DiffuserCam" = DiffuserCam(
    "/cs/student/projects1/2020/sonanuga/dataset", use_diffuser_transform=False,)
training_loader: DataLoader = DataLoader(
    diffuser_collection.train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
testing_loader: DataLoader = DataLoader(diffuser_collection.test_dataset)

device = torch.device('cuda')
PATH = './beam_prop1.pth'
simulator = Simulation(resolution=IMAGE_SIZE, batch_size=BATCH_SIZE)


class BeamPropagation(nn.Module):
    def __init__(self):
        super(BeamPropagation, self).__init__()
        self.simulator = simulator

    def forward(self, x: torch.Tensor):

        return self.simulator.diffuse_images(x.to(device))


class Net2(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=1)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 5, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 3, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=1)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 5, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(network):
    criterion = nn.MSELoss()
    optimiser = optim.Adam(network.parameters(), lr=0.001)
    transform_inputs = nn.Sequential(BeamPropagation())
    print("Training started")
    for epoch in range(5):
        running_loss = 0.0

        for i, data in enumerate(tqdm(training_loader, 0)):
            _, lenseds = data
            lenseds.to(device)
            inputs = transform_inputs(lenseds).to(device)
            lenseds = rgb_to_grayscale(lenseds).to(device)
            optimiser.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, lenseds)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if i % 25 == 24:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 24:.3f}")
                running_loss = 0.0


def test_network(network):
    for i, data in enumerate(tqdm(testing_loader, 0)):
        input, lensed = data
        input = input.to(device)
        lensed = lensed.to(device)
        output = network(input)
        print(output.shape)
        print(lensed.shape)

        image = torch.permute(torch.squeeze(output), (2, 1, 0))
        label = torch.permute(torch.squeeze(label), (2, 1, 0))

        odak.learn.tools.save_image(
            f"output{i}.jpeg", image, cmin=0, cmax=1)

        odak.learn.tools.save_image(
            f"groundtruth{i}.jpeg", label, cmin=0, cmax=1)


if __name__ == "__main__":
    net = Net()
    if torch.cuda.is_available():
        print("device: ", device)
        net.to(device)
        memory_allocated_mb = int(
            torch.cuda.max_memory_allocated(device))
        print(f"Memory allocated: {memory_allocated_mb /1024} MB")
        print("Using GPU")
    else:
        print("Using CPU")
        exit()

    torch.save(net.state_dict(), PATH)

    test_net = Net()
    test_net.load_state_dict(torch.load(PATH))

    test_network(test_net)
