import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, dim, ratio=4.0, names=("fc1", "fc2")):
        super().__init__()
        setattr(self, names[0], nn.Linear(dim, int(dim * ratio)))
        setattr(self, names[1], nn.Linear(int(dim * ratio), dim))
        self.names = names

    def forward(self, x):
        return getattr(self, self.names[1])(F.gelu(getattr(self, self.names[0])(x)))


class RelativePositionBias(nn.Module):
    def __init__(self, grid, heads):
        super().__init__()
        n = (2 * grid - 1) ** 2 + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(n, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (grid - 1)
        index = torch.zeros(grid * grid + 1, grid * grid + 1, dtype=torch.long)
        index[1:, 1:] = rel[..., 0] * (2 * grid - 1) + rel[..., 1]
        index[0, :], index[:, 0], index[0, 0] = n - 3, n - 2, n - 1
        self.register_buffer("relative_position_index", index)

    def forward(self):
        t = self.relative_position_index.shape[0]
        return self.relative_position_bias_table[self.relative_position_index.view(-1)].view(t, t, -1).permute(2, 0, 1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, bias):
        b, n, d = x.shape
        qkv = self.qkv(x) + torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
        q, k, v = qkv.reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * (d // self.heads) ** -0.5 + bias
        return self.proj((attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.gamma_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x, bias):
        a = self.attn(self.norm1(x), bias)
        m = self.mlp(self.norm2(x))
        return x + self.gamma_1 * a + self.gamma_2 * m


class USFM(nn.Module):
    size = 224

    def __init__(self, dim=768, depth=12, heads=12, patch=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.rel_pos_bias = RelativePositionBias(self.size // patch, heads)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), 1)
        bias = self.rel_pos_bias()
        for block in self.blocks:
            x = block(x, bias)
        return self.norm(x)[:, 1:].mean(1)

    def convert(self, state):
        state = {k.replace("patch_embed.proj.", "patch_embed."): v for k, v in state.items()}
        return {re.sub(r"^blocks\.\d+\.(\d+)\.", r"blocks.\1.", k): v for k, v in state.items()}


def window_partition(x, w):
    b, h, wd, c = x.shape
    ph, pw = (w - h % w) % w, (w - wd % w) % w
    x = F.pad(x, (0, 0, 0, pw, 0, ph))
    hp, wp = h + ph, wd + pw
    x = x.view(b, hp // w, w, wp // w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, w, w, c)
    return x, (hp, wp)


def window_unpartition(x, w, padded, size):
    hp, wp = padded
    b = x.shape[0] // (hp * wp // w // w)
    x = x.view(b, hp // w, wp // w, w, w, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    return x[:, : size[0], : size[1]]


def rel_pos(q, k, table):
    length = 2 * max(q, k) - 1
    if table.shape[0] != length:
        table = F.interpolate(table.reshape(1, table.shape[0], -1).permute(0, 2, 1), size=length, mode="linear").reshape(-1, length).permute(1, 0)
    qc = torch.arange(q, device=table.device)[:, None] * max(k / q, 1.0)
    kc = torch.arange(k, device=table.device)[None, :] * max(q / k, 1.0)
    return table[((qc - kc) + (k - 1) * max(q / k, 1.0)).long()]


class SAMAttention(nn.Module):
    def __init__(self, dim, heads, size):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * size - 1, dim // heads))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * size - 1, dim // heads))

    def forward(self, x):
        b, h, w, d = x.shape
        q, k, v = self.qkv(x).reshape(b, h * w, 3, self.heads, -1).permute(2, 0, 3, 1, 4).reshape(3, b * self.heads, h * w, -1).unbind(0)
        attn = (q * (d // self.heads) ** -0.5) @ k.transpose(-2, -1)
        rh, rw = rel_pos(h, h, self.rel_pos_h), rel_pos(w, w, self.rel_pos_w)
        rq = q.reshape(-1, h, w, q.shape[-1])
        attn = (attn.view(-1, h, w, h, w) + torch.einsum("bhwc,hkc->bhwk", rq, rh)[..., None] + torch.einsum("bhwc,wkc->bhwk", rq, rw)[:, :, :, None, :]).view(-1, h * w, h * w)
        x = (attn.softmax(-1) @ v).view(b, self.heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
        return self.proj(x)


class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.channel = nn.Sequential(nn.Linear(dim, dim // 4, bias=False), nn.ReLU(), nn.Linear(dim // 4, dim, bias=False), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1, bias=False), nn.ReLU(), nn.ConvTranspose2d(dim, dim, 4, 2, 1, bias=False), nn.ReLU())

    def forward(self, x):
        s = x.permute(0, 3, 1, 2)
        s = self.channel(s.mean((2, 3)))[..., None, None] * s
        s = s + self.spatial(s)
        return self.norm(s.permute(0, 2, 3, 1))


class SAMBlock(nn.Module):
    def __init__(self, dim, heads, window, grid):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SAMAttention(dim, heads, window or grid)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, names=("lin1", "lin2"))
        self.Adapter = Adapter(dim)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window:
            size = x.shape[1:3]
            x, padded = window_partition(x, self.window)
        x = self.attn(x)
        if self.window:
            x = window_unpartition(x, self.window, padded, size)
        x = shortcut + x
        n = self.norm2(x)
        return x + self.mlp(n) + self.Adapter(n)


class SAMMed2D(nn.Module):
    size = 256

    def __init__(self, dim=768, depth=12, heads=12, patch=16, window=14, global_blocks=(2, 5, 8, 11)):
        super().__init__()
        grid = self.size // patch
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid, grid, dim))
        self.blocks = nn.ModuleList(SAMBlock(dim, heads, 0 if i in global_blocks else window, grid) for i in range(depth))

    def forward(self, x):
        x = self.patch_embed.proj(x).permute(0, 2, 3, 1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return x.flatten(1, 2).mean(1)

    def convert(self, state):
        return {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")} or state


class FocalModulation(nn.Module):
    def __init__(self, dim, levels, window):
        super().__init__()
        self.levels = levels
        self.f = nn.Linear(dim, 2 * dim + levels + 1)
        self.h = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Linear(dim, dim)
        self.focal_layers = nn.ModuleList(
            nn.Sequential(nn.Conv2d(dim, dim, 2 * k + window, groups=dim, padding=(2 * k + window) // 2, bias=False), nn.GELU()) for k in range(levels)
        )

    def forward(self, x):
        c = x.shape[-1]
        q, ctx, gates = torch.split(self.f(x).permute(0, 3, 1, 2), (c, c, self.levels + 1), 1)
        total = 0
        for k, layer in enumerate(self.focal_layers):
            ctx = layer(ctx)
            total = total + ctx * gates[:, k : k + 1]
        total = total + F.gelu(ctx.mean((2, 3), keepdim=True)) * gates[:, self.levels :]
        total = total / (self.levels + 1)
        return self.proj((q * self.h(total)).permute(0, 2, 3, 1))


class FocalBlock(nn.Module):
    def __init__(self, dim, levels, window):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.modulation = FocalModulation(dim, levels, window)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.gamma_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x + self.gamma_1 * self.norm1(self.modulation(x.view(b, h, w, c)).view(b, n, c))
        return x + self.gamma_2 * self.norm2(self.mlp(x))


class PatchEmbed(nn.Module):
    def __init__(self, cin, cout, stem):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 7, 4, 2) if stem else nn.Conv2d(cin, cout, 3, 2, 1)
        self.norm = nn.LayerNorm(cout)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2:]
        return self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(-1, x.shape[1], h, w)


class FocalLayer(nn.Module):
    def __init__(self, dim, depth, levels, window, last):
        super().__init__()
        self.blocks = nn.ModuleList(FocalBlock(dim, levels, window) for _ in range(depth))
        self.downsample = None if last else PatchEmbed(dim, 2 * dim, False)


class BiomedParse(nn.Module):
    size = 224

    def __init__(self, dim=192, depths=(2, 2, 18, 2), levels=4, window=3):
        super().__init__()
        self.patch_embed = PatchEmbed(3, dim, True)
        self.layers = nn.ModuleList(FocalLayer(dim * 2**i, d, levels, window, i == len(depths) - 1) for i, d in enumerate(depths))
        for i in range(len(depths)):
            self.add_module(f"norm{i}", nn.LayerNorm(dim * 2**i))

    def forward(self, x):
        x = self.patch_embed(x)
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            for block in layer.blocks:
                x = block(x, h, w)
            if layer.downsample is not None:
                x = layer.downsample(x.transpose(1, 2).view(len(x), -1, h, w)).flatten(2).transpose(1, 2)
                h, w = (h + 1) // 2, (w + 1) // 2
        return self.norm3(x).mean(1)


class EchoCare(nn.Module):
    size = 256

    def __init__(self, dim=128):
        super().__init__()
        from monai.networks.blocks.unetr_block import UnetrBasicBlock
        from monai.networks.nets.swin_unetr import SwinTransformer

        swin = SwinTransformer(in_chans=3, embed_dim=dim, window_size=[8, 8], patch_size=[2, 2], depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], spatial_dims=2, use_v2=False)
        self.patch_embed = swin.patch_embed
        self.layers1, self.layers2, self.layers3, self.layers4 = swin.layers1, swin.layers2, swin.layers3, swin.layers4
        self.mask_token = nn.Parameter(torch.zeros(1, dim, 1, 1))
        for i in range(4):
            block = UnetrBasicBlock(spatial_dims=2, in_channels=dim * 2**i, out_channels=dim * 2**i, kernel_size=3, stride=1, norm_name="instance", res_block=True)
            setattr(self, f"layers{i + 1}c", nn.ModuleList([block]))

    def forward(self, x):
        x = self.patch_embed(x)
        for i in range(1, 5):
            x = getattr(self, f"layers{i}c")[0](x.contiguous())
            x = getattr(self, f"layers{i}")[0](x.contiguous())
        x = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],))
        return x.flatten(1, 2).mean(1)


BACKBONES = {"usfm": USFM, "sam_med2d": SAMMed2D, "biomedparse": BiomedParse, "echocare": EchoCare}


def strip(state):
    for key in ("state_dict", "model_state_dict", "model", "teacher", "student", "net"):
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    out = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "encoder.", "backbone.backbone.", "backbone."):
                if key.startswith(prefix):
                    key, changed = key[len(prefix):], True
        out[key] = value
    return out


def load(name, checkpoint):
    model = BACKBONES[name]()
    state = strip(torch.load(checkpoint, map_location="cpu", weights_only=False))
    if hasattr(model, "convert"):
        state = model.convert(state)
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing = [k for k in own if k not in state and not k.endswith(("relative_position_index", "mask_token"))]
    if missing:
        raise KeyError(f"{name}: {len(missing)} weights missing from {checkpoint}, e.g. {missing[:3]}")
    model.load_state_dict(state, strict=False)
    return model.eval()
