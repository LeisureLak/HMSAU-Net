import torch
import torch.nn as nn
import torch.nn.functional as F
# ViT module
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from einops import einops
from einops.layers.torch import Rearrange


# 位置编码
class PatchEmbeddingBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        for m, p in zip(img_size, patch_size):
            if m < p:
                raise AssertionError("patch_size should be smaller than img_size.")

        # number of patches
        self.n_patches = (
            (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        )
        # expand dim of each patch
        self.patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2] # 16*16*16*1

        # reaarange + linear projection
        self.patch_embeddings = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)', # 2, self.n_patches, self.patch_dim
                p1=patch_size[0],
                p2=patch_size[1],
                p3=patch_size[2],
            ),
            nn.Linear(self.patch_dim, hidden_size), # 2, self.n_patches, self.patch_dim -> 2, self.n_patches, hidden_size
        )
        # position encoding 1, 22*12*12, 384
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # -2～2
        self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        # random init attr
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def forward(self, x):
        x = self.patch_embeddings(x)
        embeddings = x + self.position_embeddings
        return embeddings


# MLP in ViT module
class MLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int
    ) -> None:

        super(MLP, self).__init__()

        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.ac = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.ac(x)
        x = self.linear2(x)
        return x


# SA module in ViT module
class SelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
    ) -> None:

        super(SelfAttention, self).__init__()

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # gen qkv
        self.queries = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.head_dim = hidden_size // num_heads

        # scale factor
        self.scale = self.head_dim ** -0.5

        self.rearrange = einops.rearrange

    def forward(self, x):
        q, k, v = self.rearrange(self.queries(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads)
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1) # SA matrix
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        return x


# ViT模块
class ViT_Module(nn.Module):
    def __init__(self,
                 in_channels,
                 img_size,
                 patch_size,
                 hidden_size,
                 mlp_dim,
                 num_heads,
                 ):
        super(ViT_Module, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.sa = SelfAttention(self.hidden_size, self.num_heads)
        self.mlp = MLP(self.hidden_size, self.mlp_dim)
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        x_ln = self.ln1(x)
        x = x + self.sa(x_ln)
        x_ln = self.ln2(x)
        x = x + self.mlp(x_ln)
        return x

