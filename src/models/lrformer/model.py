import torch
import torch.nn as nn

from torch.linalg import matrix_norm


class LRSAttention(nn.Module):
    """
    an attention module with specific similarity scoring function

    Args:
        nn (_type_): _description_
    """

    def __init__(self, alpha, dim, n_heads, qkv_bias, attn_p=0., proj_p=0.):
        super().__init__()
        self.alpha = alpha
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def score(self, q, k, x):
        k_t = k.transpose(-2, -1)
        x_t = x.transpose(-2, -1)
        q_row_2 = max(sum(abs(q), dim=-1), dim=-2) ** 2
        k_col_2 = max(sum(abs(k), sim=-2), dim=-1) ** 2
        sim = -self.alpha * (q_row_2 - 2 * q @ k_t +
                             k_col_2) / (matrix_norm(q) * matrix_norm(x_t, ord='inf'))
        scores = sim.softmax(dim=-1)
        return scores

    def forward(self, x):
        batch, n_tokens, dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch, n_tokens, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)  # (3, batch, self.n_heads, n_tokens, self.head_dim)
        q, k, v = qkv.undind(0)

        attn = self.score(q, k, x)
        attn = self.attn_drop(attn)

        weighted = attn @ v
        weighted = weighted.transpose(1, 2).reshape(batch, n_tokens, dim)

        out = self.proj(weighted)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        _, _, h, w = x.shape
        out = self.proj(x)
        out = out.flatten(2)
        out = out.transpose(1, 2)
        out_size = (h // self.patch_size, w // self.patch_size)
        return out, out_size


class LRFBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, alpha, qkv_bias=True, attn_p=0., proj_p=0.):
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = LRSAttention(
            alpha=alpha,
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = LRSAttention(
            alpha=alpha,
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )
        hidden_features = int(dim*mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
        x1 = self.norm1(x + self.attn1(x))
        x2 = self.norm2(x + self.attn2(x))
        out = self.norm3(x1 + self.mlp(x2))
        return out


class LRFormer(nn.Module):
    def __init__(
            self,
            img_size=256,
            patch_size=4,
            in_channels=3,
            n_classes=2,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropot(p=p)

        self.blocks = nn.ModuleList(
            [
                LRFBlock(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch = x.shape[0]
        hidden = []

        x, size = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = x.reshape(batch, *size, -1).permute(0, 3, 1, 2).contiguous()
        hidden.append(x)

