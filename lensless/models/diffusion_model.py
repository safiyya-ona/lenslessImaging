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


class DownBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, timestep_dim):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(timestep_dim, out_channels))

    def forward(self, x, emb):
        x = self.conv(x)
        x = self.pool(x)
        emb = self.emb_layer(emb)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, timestep_dim):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
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

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128, self.timestep_dim)
        # self.down_att1 = AttentionBlock(128)
        self.down2 = DownBlock(128, 256, self.timestep_dim)
        # self.down_att2 = AttentionBlock(256)
        self.down3 = DownBlock(256, 512, self.timestep_dim)
        # self.down_att3 = AttentionBlock(512)
        self.down4 = DownBlock(512, 512, self.timestep_dim)

        self.bot1 = DoubleConv(512, 1024)
        self.bot2 = DoubleConv(1024, 1024)
        self.bot3 = DoubleConv(1024, 512)

        self.up0 = UpBlock(1024, 256, self.timestep_dim)
        self.up1 = UpBlock(512, 128, self.timestep_dim)
        # self.up_att1 = AttentionBlock(128)
        self.up2 = UpBlock(256, 64, self.timestep_dim)
        # self.up_att2 = AttentionBlock(64)
        self.up3 = UpBlock(128, 64, self.timestep_dim)
        # self.up_att3 = AttentionBlock(64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def unet_forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # x2 = self.down_att1(x2)
        x3 = self.down2(x2, t)
        # x3 = self.down_att2(x3)
        x4 = self.down3(x3, t)
        # x4 = self.down_att3(x4)
        x5 = self.down4(x4, t)

        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up0(x5, x4, t)
        x = self.up1(x, x3, t)
        # x = self.up_att1(x)
        x = self.up2(x, x2, t)
        # x = self.up_att2(x)
        x = self.up3(x, x1, t)
        # x = self.up_att3(x)
        x = self.outc(x)
        return x

    def forward(self, x, t, y=None):
        emb = timestep_embedding(
            t, self.timestep_dim)
        return self.unet_forward(x, emb)
