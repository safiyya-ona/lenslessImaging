import math
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from einops import rearrange

# code adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb

UNET_STARTING_DIM = 50
RESNET_BLOCK_GROUPS = 5
RGB_CHANNELS = 3


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class PreNorm(nn.Module):
    """
    Layer that applies group normalisation before an attention block
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -embeddings
        )
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, out_channels, groups=5):
        super().__init__()
        self.proj = nn.Conv2d(dim, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Usage of residual learning as proposed by He et al. https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, out_channels, *, time_emb_dim=None, groups=5):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        self.res_conv = (
            nn.Conv2d(dim, out_channels, 1) if dim != out_channels else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """
    Layer that performs attention as proposed by Vaswani et al. https://arxiv.org/abs/1706.03762
    """

    def __init__(self, dim, heads=4, head_dims=32):
        super().__init__()
        self.scale = head_dims**-0.5
        self.heads = heads
        hidden_dim = head_dims * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Linear attention layer as implemented by Shen et al. https://github.com/lucidrains/linear-attention-transformer
    """

    def __init__(self, dim, num_heads=2, head_dims=16):
        super().__init__()
        self.scale = head_dims**-0.5
        self.heads = num_heads
        hidden_dim = head_dims * num_heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class ResiudalUNet(nn.Module):
    """
    U-Net with residual learning as proposed by He et al. https://arxiv.org/abs/1512.03385 and Ronneberger et al. https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        dim=UNET_STARTING_DIM,
        init_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=RGB_CHANNELS,
        groups=RESNET_BLOCK_GROUPS,
    ):
        super().__init__()

        assert channels == 3
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out_channels = list(zip(dims[:-1], dims[1:]))

        resnetblock_partial = partial(ResnetBlock, groups=groups)

        # timestep embeddings
        timestep_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, timestep_dim),
            nn.GELU(),
            nn.Linear(timestep_dim, timestep_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out_channels)

        for i, (in_channels, out_channels) in enumerate(in_out_channels):
            is_last = i >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        resnetblock_partial(
                            in_channels, out_channels, time_emb_dim=timestep_dim
                        ),
                        resnetblock_partial(
                            out_channels, out_channels, time_emb_dim=timestep_dim
                        ),
                        Residual(PreNorm(out_channels, LinearAttention(out_channels))),
                        Downsample(out_channels) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnetblock_partial(
            mid_dim, mid_dim, time_emb_dim=timestep_dim
        )
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = resnetblock_partial(
            mid_dim, mid_dim, time_emb_dim=timestep_dim
        )

        for ind, (in_channels, out_channels) in enumerate(
            reversed(in_out_channels[1:])
        ):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        resnetblock_partial(
                            out_channels * 2, in_channels, time_emb_dim=timestep_dim
                        ),
                        resnetblock_partial(
                            in_channels, in_channels, time_emb_dim=timestep_dim
                        ),
                        Residual(PreNorm(in_channels, LinearAttention(in_channels))),
                        Upsample(in_channels) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            resnetblock_partial(dim, dim), nn.Conv2d(dim, self.channels, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time)

        skip_connections = []

        # downsampling blocks
        for resblock1, resblock2, lin_att, downsample in self.downs:
            x = resblock1(x, t)
            x = resblock2(x, t)
            x = lin_att(x)
            skip_connections.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsampling blocks
        for resblock1, resblock2, lin_att, upsample in self.ups:
            skip = skip_connections.pop()
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat((x, skip), dim=1)
            x = resblock1(x, t)
            x = resblock2(x, t)
            x = lin_att(x)
            x = upsample(x)

        x = self.final_conv(x)
        return x
