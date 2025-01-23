import math
import torch
import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, conv_2d=False):
        super().__init__()
        if conv_2d:
            self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        else:
            self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, conv_2d=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups, conv_2d=conv_2d)
        self.block2 = Block(dim_out, dim_out, groups=groups, conv_2d=conv_2d)
        if conv_2d:
            self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        else:
            self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.conv_2d = conv_2d

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            if self.conv_2d:
                time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            else:
                time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, conv_2d=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        if conv_2d:
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
            self.to_out = nn.Sequential(
                nn.Conv2d(hidden_dim, dim, 1),
                LayerNorm(dim)
            )
        else:
            self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
            self.to_out = nn.Sequential(
                nn.Conv1d(hidden_dim, dim, 1),
                RMSNorm(dim)
            )
        self.conv_2d = conv_2d

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        if self.conv_2d:
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        else:
            q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        if self.conv_2d:
            out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=x.size(-2), y=x.size(-1))
        else:
            out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, conv_2d=False):
        super().__init__()
        self.fn = fn
        if conv_2d:
            self.norm = LayerNorm(dim)
        else:
            self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # normalize over the channel dim?
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample2d(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )


def Downsample2d(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, conv_2d=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        if conv_2d:
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        else:
            self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
            self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.conv_2d = conv_2d

    def forward(self, x):
        # b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        if self.conv_2d:
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        else:
            q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        if self.conv_2d:
            out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=x.size(-2), y=x.size(-1))
        else:
            out = rearrange(out, 'b h n d -> b (h d) n', h=self.heads)

        return self.to_out(out)


class Unet2D(nn.Module):
    def __init__(
            self,
            dim=32,
            first_output_channels=None,
            out_dim=2,
            dim_mults=(1, 2, 4, 8),
            channels=2,
            resnet_block_groups=1,
            sinusoidal_pos_emb_theta=10000,
            attn_dim_head=32,
            attn_heads=8,
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        block_klass = partial(
            ResnetBlock,
            groups=resnet_block_groups, conv_2d=True, time_emb_dim=time_dim
        )

        first_output_channels = default(first_output_channels, dim)
        self.init_conv = nn.Conv2d(input_channels, first_output_channels, kernel_size=(7, 7), padding=3)

        dims = [first_output_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(
                    dim_in,
                    LinearAttention(dim_in, conv_2d=True),
                    conv_2d=True,
                )),
                Downsample2d(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(
            mid_dim,
            Attention(mid_dim, conv_2d=True, dim_head=attn_dim_head, heads=attn_heads),
            conv_2d=True
        ))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(PreNorm(
                    dim_out, LinearAttention(dim_out, conv_2d=True),
                    conv_2d=True,
                )),
                Upsample2d(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
