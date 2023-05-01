from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# utils


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim):
    """
    Create a timestep embedding from the timestep.
    """

    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(start=0,
                            end=half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(torch.float32)

# modules


class TimestepBlock(nn.Module):
    """
    Any module where a timestep embedding is taken as a second argument
    """

    @abstractmethod
    def forward(self, x, emb):
        pass


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


class DownBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, timestep_dim):
        super().__init__()
        self.conv = DoubleConvBn(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(timestep_dim, out_channels))

    def forward(self, x, emb):
        x = self.conv(x)
        x_pool = self.pool(x)
        emb = self.emb_layer(emb)[:, :, None, None].repeat(
            1, 1, x_pool.shape[-2], x_pool.shape[-1])
        return x, x_pool + emb


class UpBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, timestep_dim):
        super().__init__()
        self.conv = nn.Sequential(SingleConvBn(
            in_channels*2, in_channels), DoubleConvBn(in_channels, out_channels))
        self.emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(timestep_dim, out_channels))

    def forward(self, x, skip, emb):
        x = F.interpolate(x, scale_factor=2)
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(emb)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight,
                         v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, timestep_dim=32, num_classes=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.timestep_dim = timestep_dim
        self.time_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, timestep_dim),
        )
        if num_classes is not None:
            self.label_embed = nn.Embedding(num_classes, timestep_dim)

        self.down1 = DownBlock(in_channels, 24, self.timestep_dim)
        self.down2 = DownBlock(24, 64, self.timestep_dim)
        self.down3 = DownBlock(64, 128, self.timestep_dim)
        self.down4 = DownBlock(128, 256, self.timestep_dim)
        self.down5 = DownBlock(256, 512, self.timestep_dim)

        self.bot1 = DoubleConv(512, 512)

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

        x = self.bot1(x)

        x = self.up5(x, x5, t)
        x = self.up4(x, x4, t)
        x = self.up3(x, x3, t)
        x = self.up2(x, x2, t)
        x = self.up1(x, x1, t)

        x = self.outc(x)
        return x

    def forward(self, x, t, y=None):
        emb = timestep_embedding(
            t, self.timestep_dim)
        return self.unet_forward(x, emb)
