import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_unet import DoubleConv

# UNet implementation from Otkay et al. https://arxiv.org/abs/1804.03999


class AttentionGateBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None) -> None:
        super(AttentionGateBlock, self).__init__()

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # W which is applied to both g (gating signal) and x_l (input features) for the additive attention gate
        self.W = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.in_channels),
        )

        # theta is applied to the input features x_l
        self.theta = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # phi is applied to the gating signal g
        self.phi = nn.Conv2d(
            in_channels=self.gating_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # psi is applied to the output of the attention gate
        self.psi = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0), "g and x_l must have the same batch size"

        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = F.relu(theta_x + phi_g)

        sigmoid_psi_f = torch.sigmoid(self.psi(f))

        up_sigmoid_psi_f = F.interpolate(
            sigmoid_psi_f, size=input_size[2:], mode="bilinear", align_corners=True
        )

        y = up_sigmoid_psi_f * x
        return self.W(y)


class AttentionUNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, features=[64, 128, 256, 512, 1024]
    ) -> None:
        super(AttentionUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.attention_blocks.append(
                AttentionGateBlock(
                    in_channels=feature,
                    gating_channels=feature,
                    inter_channels=feature // 2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            skip_connection = self.attention_blocks[idx // 2](x, skip_connection)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
