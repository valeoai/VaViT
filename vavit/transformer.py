# Copyright 2026 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def apply_2d_rope(q, k, coords, inv_freq):
    # q, k: batch_size x num_heads x sequence_length x dim_head
    # coords: (x, y) of size batch_size x sequence_length x 2

    # Extract x and y
    t_x = coords[..., 0:1].unsqueeze(1) * inv_freq
    t_y = coords[..., 1:2].unsqueeze(1) * inv_freq

    # Generate sin / cos for both axes
    cos_x, sin_x = t_x.cos(), t_x.sin()
    cos_y, sin_y = t_y.cos(), t_y.sin()

    # Split q and k into two: one for x axis, one for y axis
    q_x, q_y = q.chunk(2, dim=-1)
    k_x, k_y = k.chunk(2, dim=-1)

    # Apply rotation
    q = torch.cat([rotate(q_x, cos_x, sin_x), rotate(q_y, cos_y, sin_y)], dim=-1)
    k = torch.cat([rotate(k_x, cos_x, sin_x), rotate(k_y, cos_y, sin_y)], dim=-1)

    return q, k


class DropPath(nn.Module):
    def __init__(self, fn, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob
        self.scale = 1.0 / self.keep_prob

        # Function on which to apply droppath
        self.fn = fn

    def extra_repr(self):
        return f"prob={self.drop_prob:.3f}"

    def forward(self, x, coords=None):
        if not self.training or self.drop_prob == 0.0:
            return x + self.fn(x, coords)

        # Mask batch entries
        batch_size = x.shape[0]
        keep_batch = torch.bernoulli(
            torch.full((batch_size,), self.keep_prob, device=x.device)
        ).bool()

        # Drop everything
        if not keep_batch.any():
            return x

        # Efficient droppath
        idx_keep = torch.nonzero(keep_batch).flatten()
        if coords is not None:
            coords = coords[idx_keep]
        residual = self.fn(x[idx_keep], coords).to(x.dtype)

        # Add residual
        out = x.clone()
        out.index_add_(0, idx_keep, residual, alpha=self.scale)

        return out


class ChannelMix(nn.Module):
    def __init__(
        self,
        dim,
        expansion,
        drop_path_prob,
        layerscale_init=1e-5,
    ):
        super().__init__()

        # Pre-norm
        self.norm = nn.LayerNorm(dim)

        # MLP
        hidden_dim = int(dim * expansion)
        self.l1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, dim, bias=False)

        # LayerScale
        self.gamma = nn.Parameter(
            layerscale_init * torch.ones((dim)),
            requires_grad=True,
        )

        # DropPath wrapper
        self.drop_path = DropPath(self._forward_, drop_path_prob)

        # Init.
        self.init_weights()

    def init_weights(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.l1.weight, std=0.02)
        nn.init.trunc_normal_(self.l2.weight, std=0.02)

    def _forward_(self, x, *args, **kwargs):
        x = self.norm(x)
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x * self.gamma

    def forward(self, tokens):
        # Drop path calls _forward_
        return self.drop_path(tokens)


class SpatialMix(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        drop_path_prob,
        layerscale_init=1e-5,
        rope_freq=10000,
    ):
        super().__init__()

        # Dimensions
        self.num_heads = num_heads
        self.dim_head = dim_head = dim // num_heads
        self.scale = dim_head**-0.5

        # Pre-norm
        self.norm = nn.LayerNorm(dim)

        # Attention
        self.qkv = nn.Linear(dim, 3 * num_heads * dim_head, bias=False)
        self.proj = nn.Linear(num_heads * dim_head, dim, bias=False)

        # RoPE
        half_head = dim_head // 2
        self.register_buffer(
            "inv_freq",
            1.0 / (rope_freq ** (torch.arange(0, half_head, 2).float() / half_head)),
        )

        # LayerScale
        self.gamma = nn.Parameter(
            layerscale_init * torch.ones((dim)),
            requires_grad=True,
        )

        # DropPath wrapper
        self.drop_path = DropPath(self._forward_, drop_path_prob)

        # Init.
        self.init_weights()

    def init_weights(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def _forward_(self, x, coords):
        # Shape
        B, N, C = x.shape

        # Norm
        x_norm = self.norm(x)

        # Extract qkv
        qkv = self.qkv(x_norm).view(B, N, 3, self.num_heads, self.dim_head)
        q, k, v = torch.unbind(qkv, dim=2)

        # B x num_heads x sequence_length x dim_head
        q, k, v = [t.transpose(1, 2).contiguous() for t in [q, k, v]]

        # PE
        q, k = apply_2d_rope(q, k, coords, self.inv_freq)

        # Attention
        x_attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            scale=self.scale,
        )

        # Projection
        x_attn = x_attn.transpose(1, 2)
        x_attn = x_attn.reshape(B, N, self.num_heads * self.dim_head)
        out = self.proj(x_attn)

        # Layerscale
        return out * self.gamma

    def forward(self, tokens, coords):
        # Drop path calls compute_attention
        return self.drop_path(tokens, coords)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        expansion,
        drop_path,
    ):
        super().__init__()

        self.channel_mix = nn.ModuleList(
            [ChannelMix(dim, expansion, drop_path) for i in range(depth)]
        )

        self.spatial_mix = nn.ModuleList(
            [SpatialMix(dim, num_heads, drop_path) for i in range(depth)]
        )

    def forward(self, tokens, rope_coords):
        for i, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            tokens = smix(tokens, rope_coords)
            tokens = cmix(tokens)
        return tokens
