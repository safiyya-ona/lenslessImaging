from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where a timestep embedding is taken as a second argument
    """

    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, encoder_out=None, mask=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, encoder_out, mask=mask)
            else:
                x = layer(x)
        return x


class SingleConv(nn.Module):
    """
    Applied on upsampling and downsampling blocks
    """

    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


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


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(DownBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = SingleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(UpBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = SingleConv(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x


class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, emb_channels, out_channels=None, up=False, down=False):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.up_down = up or down

        if up:
            self.h_upd = UpBlock(in_channels)
            self.x_upd = UpBlock(in_channels)
        elif down:
            self.h_upd = DownBlock(in_channels)
            self.x_upd = DownBlock(in_channels)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        print(in_channels)
        self.in_layers = nn.Sequential(nn.GroupNorm(8, self.in_channels), nn.SiLU(
        ), nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1))

        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(self.emb_channels, self.out_channels))

        self.out_layers = nn.Sequential(nn.GroupNorm(8, self.out_channels), nn.SiLU(
        ), zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)))

        if self.in_channels == self.out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.up_down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        encoder_channels=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = nn.Conv1d(encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, encoder_out=None, mask=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = self.attention(qkv, encoder_out, mask=mask)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None, mask=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3,
                              length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(
                bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        if mask is not None:
            mask = F.pad(mask, (0, length), value=0.0)
            mask = (
                mask.unsqueeze(1)
                .expand(-1, self.n_heads, -1)
                .reshape(bs * self.n_heads, 1, -1)
            )
            weight = weight + mask
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down_att1 = AttentionBlock(128)
        self.down2 = DownBlock(128, 256)
        self.down_att2 = AttentionBlock(256)
        self.down3 = DownBlock(256, 512)
        self.down_att3 = AttentionBlock(512)

        self.up1 = UpBlock(512, 256)
        self.up_att1 = AttentionBlock(256)
        self.up2 = UpBlock(256, 128)
        self.up_att2 = AttentionBlock(128)
        self.up3 = UpBlock(128, 64)
        self.up_att3 = AttentionBlock(64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def unet_forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.down_att1(x2)
        x3 = self.down2(x2, t)
        x3 = self.down_att2(x3)
        x4 = self.down3(x3, t)
        x4 = self.down_att3(x4)

        x = self.up1(x4, x3, t)
        x = self.up_att1(x)
        x = self.up2(x, x2, t)
        x = self.up_att2(x)
        x = self.up3(x, x1)
        x = self.up_att3(x)
        x = self.outc(x)
        return x

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(
            timesteps, self.time_embed))

        return self.unet_forward(x, emb)


class ResUNet(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks=2, attention_resolutions=[2, 1], num_heads=1, num_head_channels=16, channel_mult=(1, 2, 4, 8)):
        super(ResUNet, self).__init__()
        self.model_channels = model_channels
        # not understood yet
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(
                    ch, time_embed_dim, int(mult * model_channels))]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(
                        ch, num_heads, num_head_channels))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(ResidualBlock(ch, time_embed_dim, ch, down=True)))
                ch = out_ch
                input_block_chans.append(ch)
                ds += 2
                self._feature_size += ch

        # middle block
        self.middle_block = TimestepEmbedSequential(ResidualBlock(
            ch, time_embed_dim), (AttentionBlock(ch, num_heads, num_head_channels)))

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                input_block_channel = input_block_chans.pop()
                layers = [ResidualBlock(
                    ch + input_block_channel, time_embed_dim, int(mult * model_channels))]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(
                        ch, num_heads, num_head_channels))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResidualBlock(
                        ch, time_embed_dim, ch, up=True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.Identity(
        ), zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)))

    def forward(self, x, timesteps, y=None):

        hs = []

        emb = self.time_embed(timestep_embedding(
            timesteps, self.model_channels))

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)
