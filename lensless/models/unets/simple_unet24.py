import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SingleConvBn(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvBn(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBn(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_pool = self.pool(x)
        return x, x_pool


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestep_dim):
        super().__init__()
        self.conv = nn.Sequential(SingleConvBn(
            in_channels*2, in_channels), DoubleConvBn(in_channels, out_channels))
        self.emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(timestep_dim, out_channels))

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2)
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down1 = DownBlock(in_channels, 24, self.timestep_dim)
        self.down2 = DownBlock(24, 64, self.timestep_dim)
        self.down3 = DownBlock(64, 128, self.timestep_dim)
        self.down4 = DownBlock(128, 256, self.timestep_dim)
        self.down5 = DownBlock(256, 512, self.timestep_dim)
        self.down6 = DownBlock(512, 1024, self.timestep_dim)

        self.bot1 = DoubleConv(1024, 1024)

        self.up6 = UpBlock(1024, 512, self.timestep_dim)
        self.up5 = UpBlock(512, 256, self.timestep_dim)
        self.up4 = UpBlock(256, 128, self.timestep_dim)
        self.up3 = UpBlock(128, 64, self.timestep_dim)
        self.up2 = UpBlock(64, 24, self.timestep_dim)
        self.up1 = UpBlock(24, 24, self.timestep_dim)
        self.outc = nn.Conv2d(24, out_channels, 1)

    def unet_forward(self, x, t):
        x1, x = self.down1(x, t)
        x2, x = self.down2(x, t)
        x3, x = self.down3(x, t)
        x4, x = self.down4(x, t)
        x5, x = self.down5(x, t)
        x6, x = self.down6(x, t)

        x = self.bot1(x)

        x = self.up6(x, x6, t)
        x = self.up5(x, x5, t)
        x = self.up4(x, x4, t)
        x = self.up3(x, x3, t)
        x = self.up2(x, x2, t)
        x = self.up1(x, x1, t)

        x = self.outc(x)
        return x

    def forward(self, x):
        return self.unet_forward(x)
