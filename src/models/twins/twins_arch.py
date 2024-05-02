""" Twins
A PyTorch impl of : `Twins: Revisiting the Design of Spatial Attention in Vision Transformers`
    - https://arxiv.org/pdf/2104.13840.pdf

Code/weights from https://github.com/Meituan-AutoML/Twins, original copyright/license info below

"""
# --------------------------------------------------------
# Twins
# Copyright (c) 2021 Meituan
# Licensed under The Apache 2.0 License [see LICENSE for details]
# Written by Xinjie Li, Xiangxiang Chu
# --------------------------------------------------------
import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, DropPath, to_2tuple, trunc_normal_, use_fused_attn

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 2, 'input_size': (3, 256, 256), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embeds.0.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'twins_pcpvt_small': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_small-e70e7e7a.pth',
        ),
    'twins_pcpvt_base': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_base-e5ecb09b.pth',
        ),
    'twins_pcpvt_large': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_large-d273f802.pth',
        ),
    'twins_svt_small': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_small-42e5f78c.pth',
        ),
    'twins_svt_base': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_base-c2265010.pth',
        ),
    'twins_svt_large': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_large-90f6aaa9.pth',
        ),
}

Size_ = Tuple[int, int]


class Attention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(
            B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, drop)
        elif ws == 1:
            self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        else:
            self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Size_):
        x = x + self.drop_path(self.attn(self.norm1(x), size))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim), )
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, Size_]:
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        out_size = (H // self.patch_size[0], W // self.patch_size[1])

        return x, out_size


class Twins(nn.Module):
    """ Twins Vision Transfomer (Revisiting Spatial Attention)

    Adapted from PVT (PyramidVisionTransformer) class at https://github.com/whai362/PVT.git
    """
    def __init__(
            self, img_size=256, patch_size=4, in_chans=3, num_classes=2, global_pool='avg',
            embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), depths=(3, 4, 6, 3),
            sr_ratios=(8, 4, 2, 1), wss=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.grad_checkpointing = False

        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])

        self.norm = norm_layer(self.num_features)

        # init weights
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embeds.0',  # stem and embed
            blocks=[
                (r'^(?:blocks|patch_embeds|pos_block)\.(\d+)', None),
                ('^norm', (99999,))
            ] if coarse else [
                (r'^blocks\.(\d+)\.(\d+)', None),
                (r'^(?:patch_embeds|pos_block)\.(\d+)', (0,)),
                (r'^norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reshape_hidden(self, hidden):
        pass

    def forward_features(self, x):
        B = x.shape[0]
        hidden = []
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            hidden.append(x)
        return x, hidden

    def forward(self, x):
        x, hidden = self.forward_features(x)
        return x, hidden