import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rgb_to_grayscale, resize

from lensless.helpers.diffusercam import DiffuserCam
from lensless.helpers.beam_propagation import Simulation

BATCH_SIZE = 10
IMAGE_SIZE = 270, 480


device = torch.device('cuda')


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),  # no need for bias with batch norm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256, 512, 1024]) -> None:
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()  # want to be able to do model eval for batch norm
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down part of the network
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # up part of the network
        for feature in reversed(features):
            self.ups.append(
                # doubles height and width of image
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))  # up and two convs

        # features -1 for 512m, and *2 for the up
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # 1x1 conv to get to 1 channel
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # reverse the skip connections
        skip_connections = skip_connections[::-1]

        # want to do the up and double conv in pairs therefore step of 2
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # want to take every one
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # resize to the size of the skip connection to image height and width
                # x = resize(x, size=skip_connection.shape[2:])
                x = F.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            # concat along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape, x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
